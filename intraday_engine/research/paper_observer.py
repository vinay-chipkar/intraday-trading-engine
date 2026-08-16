from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import settings
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.scanner.service import ScannerConfig, scan_top10
from intraday_engine.signals.engine import SignalConfig, TradeSignal, generate_signal
from intraday_engine.market.session import is_session_open
from intraday_engine.storage.db import conn
from intraday_engine.strategy.point_in_time import enrich_point_in_time
from intraday_engine.technical.feature_engine import snapshot_from_row
from intraday_engine.versioning import FEATURE_ENGINE_VERSION, STRATEGY_VERSION, get_code_commit

IST = ZoneInfo("Asia/Kolkata")

# How old the latest bar is allowed to be, in minutes, before an intraday
# (same trading day) signal is treated as stale rather than merely "the
# usual couple of minutes of ingestion lag". paper_session.py ticks every 5
# minutes by default, so 15 minutes means ~2-3 consecutive ticks' worth of
# ingestion have failed to advance the data at all -- long enough that this
# is a genuine pipeline problem, not routine per-tick latency.
MAX_INTRADAY_BAR_AGE_MINUTES = 15


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


def is_bar_stale_intraday(
    bar_time: object, now: object, max_age_minutes: int = MAX_INTRADAY_BAR_AGE_MINUTES
) -> bool:
    """True if `bar_time` is from *today* but old enough relative to `now`
    that ingestion has clearly stalled mid-session -- e.g. an outage that
    started 20 minutes ago but hasn't yet rolled over into a new trading day,
    so is_bar_stale's day-boundary check alone would miss it. A signal built
    from a bar this old is stale in effect even though it technically isn't
    "yesterday's" data.
    """
    bar_ts = pd.Timestamp(bar_time)
    now_ts = pd.Timestamp(now)
    if bar_ts.tzinfo is None or now_ts.tzinfo is None:
        return False
    age_minutes = (now_ts - bar_ts).total_seconds() / 60.0
    return age_minutes > max_age_minutes


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
        # Provenance for new rows only (see intraday_engine/versioning.py) --
        # historical rows predate these columns and stay NULL rather than
        # being backfilled with a version they weren't actually generated with.
        connection.execute("ALTER TABLE paper_observations ADD COLUMN IF NOT EXISTS strategy_version VARCHAR")
        connection.execute("ALTER TABLE paper_observations ADD COLUMN IF NOT EXISTS feature_engine_version VARCHAR")
        connection.execute("ALTER TABLE paper_observations ADD COLUMN IF NOT EXISTS code_commit VARCHAR")
        # AVAILABLE | MARKET_CONTEXT_MISSING -- see _latest_context(). Keeps a
        # missing premarket capture distinguishable from a genuinely neutral
        # market (regime='NEUTRAL', a real string) rather than both looking
        # like market_regime=None/market_score=0.0.
        connection.execute(
            "ALTER TABLE paper_observations ADD COLUMN IF NOT EXISTS market_context_status VARCHAR"
        )
        # NULL | PRIOR_DAY | INTRADAY_STALE -- see is_bar_stale/is_bar_stale_intraday.
        # Distinguishes *why* status=STALE_DATA fired, for diagnostics.
        connection.execute("ALTER TABLE paper_observations ADD COLUMN IF NOT EXISTS stale_reason VARCHAR")
        # The exact point-in-time feature vector generate_signal used for this
        # observation (technical/feature_engine.py::snapshot_from_row, JSON),
        # captured at signal time -- not recomputed later from candles that
        # may since have been revised (see market/ingestion.py::upsert_candles).
        connection.execute(
            "ALTER TABLE paper_observations ADD COLUMN IF NOT EXISTS decision_features VARCHAR"
        )
    finally:
        connection.close()


def _json_tuple(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _decision_observation_id(
    *, instrument_key: str, bar_time: object, strategy_version: str | None, feature_engine_version: str | None
) -> str:
    """Deterministic observation_id for one (instrument, decision bar, code
    version) triple -- the idempotency key that makes observe_once() safe to
    replay. A workflow tick re-run against a bar that already produced an
    observation under the same strategy/feature-engine version hashes to the
    exact same id, so persist_observations()'s INSERT OR IGNORE silently
    drops the replay instead of recording a second, duplicate economic
    observation for a decision that was already made. A genuine code version
    bump (strategy_version/feature_engine_version changes) deliberately
    produces a *different* id -- that is a new decision, not a replay, and
    must get its own row."""
    key = f"{instrument_key}|{pd.Timestamp(bar_time).isoformat()}|{strategy_version}|{feature_engine_version}"
    return f"obs-{hashlib.sha256(key.encode()).hexdigest()[:32]}"


def build_observation(
    *,
    observed_at: datetime,
    trading_date: date,
    candidate: dict,
    signal: TradeSignal,
    market_regime: str | None,
    market_score: float | None,
    bar_time: object,
    market_context_status: str = "AVAILABLE",
    stale: bool = False,
    stale_reason: str | None = None,
    decision_features: dict | None = None,
) -> dict:
    action = signal.action
    # A stale bar means the signal was computed from a prior trading day's data
    # (PRIOR_DAY) or from a same-day bar old enough that ingestion has clearly
    # stalled mid-session (INTRADAY_STALE) -- either way it must never be
    # surfaced as an actionable trade plan, regardless of what action/score
    # generate_signal happened to return for that stale input.
    stale = stale or stale_reason is not None
    actionable = action in {"BUY", "SELL"} and not stale
    if stale:
        status = "STALE_DATA"
    elif actionable:
        status = "SIGNAL_PENDING"
    else:
        status = "NO_SIGNAL"
    return {
        "observation_id": _decision_observation_id(
            instrument_key=candidate["instrument_key"],
            bar_time=bar_time,
            strategy_version=STRATEGY_VERSION,
            feature_engine_version=FEATURE_ENGINE_VERSION,
        ),
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
        "strategy_version": STRATEGY_VERSION,
        "feature_engine_version": FEATURE_ENGINE_VERSION,
        "code_commit": get_code_commit(),
        "market_context_status": market_context_status,
        "stale_reason": stale_reason,
        "decision_features": json.dumps(decision_features, default=str) if decision_features is not None else None,
    }


def _load_bars(instrument_key: str) -> pd.DataFrame:
    """Filtered by instrument_key, not symbol: if a symbol is ever remapped
    to a different instrument_key (a relisting/rename -- SYMBOL_ALIASES
    already exists in market/upstox.py because this has happened before),
    a symbol-only query would silently splice two different securities'
    candle histories together. instrument_key is the stable identifier."""
    connection = conn()
    try:
        return connection.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE instrument_key = ? AND interval = '1m'
            ORDER BY timestamp
            """,
            [instrument_key],
        ).df()
    finally:
        connection.close()


def _signal_for_symbol(
    symbol: str,
    *,
    instrument_key: str,
    market_score: float,
    min_score: float,
) -> tuple[TradeSignal, object, dict]:
    bars = _load_bars(instrument_key)
    if bars.empty:
        raise LookupError(f"No 1m candles stored for {symbol} ({instrument_key})")
    features = enrich_point_in_time(bars)
    row = features.iloc[-1]
    signal = generate_signal(
        row.to_dict(),
        market_score=market_score,
        config=SignalConfig(buy_threshold=min_score, sell_threshold=-min_score),
        symbol=symbol,
        event_time=row["timestamp"],
    )
    # The exact point-in-time feature vector generate_signal just decided
    # from -- captured here, not recomputed later, so diagnostics/learning
    # data reflect the true decision-time inputs even if the underlying
    # candles get revised afterward (see market/ingestion.py::upsert_candles)
    # or feature_engine.py's independent re-enrichment would compute
    # something slightly different.
    decision_features = snapshot_from_row(row, symbol=symbol, instrument_key=instrument_key)
    return signal, row["timestamp"], decision_features


def persist_observations(rows: list[dict]) -> int:
    """Insert new observations, silently dropping any whose observation_id
    (the (instrument_key, bar_time, strategy_version, feature_engine_version)
    idempotency key -- see _decision_observation_id) already exists. This is
    what makes a replayed workflow tick a no-op instead of a duplicate
    economic observation: returns the count actually inserted, which can be
    less than len(rows) on a replay."""
    if not rows:
        return 0
    ensure_observation_table()
    connection = conn()
    try:
        frame = pd.DataFrame(rows)
        before = connection.execute("SELECT COUNT(*) FROM paper_observations").fetchone()[0]
        connection.register("paper_observations_in", frame)
        connection.execute(
            "INSERT OR IGNORE INTO paper_observations SELECT * FROM paper_observations_in"
        )
        after = connection.execute("SELECT COUNT(*) FROM paper_observations").fetchone()[0]
        return after - before
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
    context = _latest_context(trading_date)
    market_score = float(context.get("score") or 0.0)
    candidates = scan_top10(
        UpstoxREST(),
        ScannerConfig(limit=limit, news_lookback_hours=settings.news_lookback_hours),
        trading_date=trading_date,
    )

    rows: list[dict] = []
    for candidate in candidates:
        signal, bar_time, decision_features = _signal_for_symbol(
            candidate["symbol"],
            instrument_key=candidate["instrument_key"],
            market_score=market_score,
            min_score=min_score,
        )
        if is_bar_stale(bar_time, trading_date):
            stale_reason = "PRIOR_DAY"
        elif is_session_open(observed_at) and is_bar_stale_intraday(bar_time, observed_at):
            stale_reason = "INTRADAY_STALE"
        else:
            stale_reason = None
        rows.append(
            build_observation(
                observed_at=observed_at,
                trading_date=trading_date,
                candidate=candidate,
                signal=signal,
                market_regime=context.get("regime"),
                market_score=market_score,
                market_context_status=context.get("status", "AVAILABLE"),
                bar_time=bar_time,
                stale_reason=stale_reason,
                decision_features=decision_features,
            )
        )
    persist_observations(rows)
    return rows


def _latest_context(trading_date: date) -> dict:
    """Today's premarket snapshot only -- never a prior day's, even if it's
    the most recently captured row (see storage/db.py::latest_market_context
    for why an unscoped "most recent" query is a point-in-time integrity
    risk here).

    When no row exists for today, the *signal engine's* market_score input
    still falls back to the same neutral 0.0 it always has (unchanged
    scoring behavior) -- but the returned "status" is MARKET_CONTEXT_MISSING,
    never AVAILABLE, so a caller can tell "context genuinely came back
    neutral" apart from "context wasn't captured at all" instead of both
    looking like an ordinary neutral market.
    """
    connection = conn()
    try:
        row = connection.execute(
            "SELECT regime, score FROM market_context WHERE trading_date = ? "
            "ORDER BY captured_at DESC LIMIT 1",
            [trading_date],
        ).fetchone()
        if row is None:
            return {"regime": None, "score": 0.0, "status": "MARKET_CONTEXT_MISSING"}
        return {"regime": row[0], "score": row[1], "status": "AVAILABLE"}
    finally:
        connection.close()
