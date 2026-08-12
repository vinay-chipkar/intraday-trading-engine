from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import settings
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.scanner.service import ScannerConfig, scan_top10
from intraday_engine.signals.engine import SignalConfig, TradeSignal, generate_signal
from intraday_engine.storage.db import conn
from intraday_engine.strategy.point_in_time import enrich_point_in_time

IST = ZoneInfo("Asia/Kolkata")


def is_bar_stale(bar_time: object, trading_date: date) -> bool:
    """True if the point-in-time bar used for a signal predates the trading day.

    The scanner scores candidates from live Upstox quotes, but the signal engine
    is fed from locally-stored 1m candles (`_load_bars`) with no freshness check
    of its own -- if intraday ingestion for `trading_date` hasn't run yet (or
    failed), `_signal_for_symbol` silently scores the *previous* trading day's
    last bar as if it were "now", against a candidate whose score/RVOL/momentum
    are all from today. That produces a signal that looks like a legitimate
    rejection of today's setup but is actually blind to today's price action
    entirely.
    """
    ts = pd.Timestamp(bar_time)
    bar_date = ts.tz_convert(IST).date() if ts.tzinfo is not None else ts.date()
    return bar_date < trading_date


OBSERVATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_observations(
    observation_id VARCHAR PRIMARY KEY,
    observed_at TIMESTAMPTZ,
    bar_time TIMESTAMPTZ,
    trading_date DATE,
    symbol VARCHAR,
    instrument_key VARCHAR,
    scanner_rank INTEGER,
    candidate_score DOUBLE,
    price_change_pct DOUBLE,
    relative_volume DOUBLE,
    vwap DOUBLE,
    market_regime VARCHAR,
    market_score DOUBLE,
    signal_action VARCHAR,
    signal_score DOUBLE,
    confidence DOUBLE,
    entry_price DOUBLE,
    stop_loss DOUBLE,
    target DOUBLE,
    signal_reasons VARCHAR,
    signal_blockers VARCHAR,
    status VARCHAR
);
"""


def ensure_observation_table(path: str | None = None) -> None:
    connection = conn(path)
    try:
        connection.execute(OBSERVATION_SCHEMA)
    finally:
        connection.close()


def _json_tuple(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def build_observation(
    *,
    observed_at: datetime,
    trading_date: date,
    candidate: dict,
    signal: TradeSignal,
    market_regime: str | None,
    market_score: float | None,
    bar_time: object,
    stale: bool = False,
) -> dict:
    action = signal.action
    # A stale bar means the signal was computed from a prior trading day's data,
    # not today's -- it must never be surfaced as an actionable trade plan (see
    # is_bar_stale for why), regardless of what action/score generate_signal
    # happened to return for that (wrong-day) input.
    actionable = action in {"BUY", "SELL"} and not stale
    if stale:
        status = "STALE_DATA"
    elif actionable:
        status = "SIGNAL_PENDING"
    else:
        status = "NO_SIGNAL"
    return {
        "observation_id": f"obs-{uuid.uuid4().hex}",
        "observed_at": observed_at,
        "bar_time": bar_time,
        "trading_date": trading_date,
        "symbol": candidate["symbol"],
        "instrument_key": candidate["instrument_key"],
        "scanner_rank": int(candidate.get("rank") or 0),
        "candidate_score": float(candidate.get("candidate_score") or 0.0),
        "price_change_pct": candidate.get("change_pct"),
        "relative_volume": candidate.get("relative_volume"),
        "vwap": candidate.get("vwap"),
        "market_regime": market_regime,
        "market_score": market_score,
        "signal_action": action,
        "signal_score": float(signal.score),
        "confidence": float(signal.confidence),
        "entry_price": signal.entry if actionable else None,
        "stop_loss": signal.stop_loss if actionable else None,
        "target": signal.target if actionable else None,
        "signal_reasons": _json_tuple(signal.reasons),
        "signal_blockers": _json_tuple(signal.blockers),
        "status": status,
    }


def _load_bars(symbol: str) -> pd.DataFrame:
    connection = conn()
    try:
        return connection.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = '1m'
            ORDER BY timestamp
            """,
            [symbol],
        ).df()
    finally:
        connection.close()


def _signal_for_symbol(
    symbol: str,
    *,
    market_score: float,
    min_score: float,
) -> tuple[TradeSignal, object]:
    bars = _load_bars(symbol)
    if bars.empty:
        raise LookupError(f"No 1m candles stored for {symbol}")
    features = enrich_point_in_time(bars)
    row = features.iloc[-1]
    signal = generate_signal(
        row.to_dict(),
        market_score=market_score,
        config=SignalConfig(buy_threshold=min_score, sell_threshold=-min_score),
        symbol=symbol,
        event_time=row["timestamp"],
    )
    return signal, row["timestamp"]


def persist_observations(rows: list[dict]) -> int:
    if not rows:
        return 0
    ensure_observation_table()
    connection = conn()
    try:
        frame = pd.DataFrame(rows)
        connection.register("paper_observations_in", frame)
        connection.execute(
            "INSERT INTO paper_observations SELECT * FROM paper_observations_in"
        )
        return len(frame)
    finally:
        connection.unregister("paper_observations_in")
        connection.close()


def observe_once(
    *,
    limit: int = 10,
    min_score: float = 60.0,
    trading_date: date | None = None,
) -> list[dict]:
    observed_at = datetime.now().astimezone()
    trading_date = trading_date or observed_at.date()
    context = _latest_context()
    market_score = float(context.get("score") or 0.0)
    candidates = scan_top10(
        UpstoxREST(),
        ScannerConfig(limit=limit, news_lookback_hours=settings.news_lookback_hours),
        trading_date=trading_date,
    )

    rows: list[dict] = []
    for candidate in candidates:
        signal, bar_time = _signal_for_symbol(
            candidate["symbol"], market_score=market_score, min_score=min_score
        )
        rows.append(
            build_observation(
                observed_at=observed_at,
                trading_date=trading_date,
                candidate=candidate,
                signal=signal,
                market_regime=context.get("regime"),
                market_score=market_score,
                bar_time=bar_time,
                stale=is_bar_stale(bar_time, trading_date),
            )
        )
    persist_observations(rows)
    return rows


def _latest_context() -> dict:
    connection = conn()
    try:
        row = connection.execute(
            "SELECT regime, score FROM market_context ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return {"regime": None, "score": 0.0}
        return {"regime": row[0], "score": row[1]}
    finally:
        connection.close()
