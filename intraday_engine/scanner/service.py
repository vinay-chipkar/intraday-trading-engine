from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import conn, insert_df, latest_market_score, latest_symbol_news_scores
from .ranking import rank_candidates

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


@dataclass(frozen=True)
class ScannerConfig:
    limit: int = 10
    lookback_days: int = 20
    minimum_price: float = 50.0
    minimum_avg_daily_traded_value: float = 50_000_000.0
    news_lookback_hours: int = 24


def _elapsed_minutes(timestamp: datetime) -> float:
    local = timestamp.astimezone(IST)
    start = datetime.combine(local.date(), SESSION_OPEN, tzinfo=IST)
    end = datetime.combine(local.date(), SESSION_CLOSE, tzinfo=IST)
    if local <= start:
        return 1.0
    return max(1.0, min((min(local, end) - start).total_seconds() / 60.0, 375.0))


def _historical_liquidity(symbols: list[str], trading_date: date, lookback_days: int) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    connection = conn()
    try:
        placeholders = ",".join("?" for _ in symbols)
        query = f"""
        WITH daily AS (
            SELECT
                symbol,
                CAST(timestamp AT TIME ZONE 'Asia/Kolkata' AS DATE) AS trading_date,
                SUM(volume) AS daily_volume,
                SUM(volume * close) AS daily_traded_value
            FROM candles
            WHERE interval = '1m'
              AND symbol IN ({placeholders})
              AND CAST(timestamp AT TIME ZONE 'Asia/Kolkata' AS DATE) < ?
            GROUP BY symbol, CAST(timestamp AT TIME ZONE 'Asia/Kolkata' AS DATE)
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trading_date DESC) AS rn
            FROM daily
        )
        SELECT
            symbol,
            AVG(daily_volume) AS avg_daily_volume,
            AVG(daily_traded_value) AS avg_daily_traded_value,
            COUNT(*) AS history_days
        FROM ranked
        WHERE rn <= ?
        GROUP BY symbol
        """
        return connection.execute(query, [*symbols, trading_date, lookback_days]).df()
    finally:
        connection.close()


def _current_day_vwap(symbols: list[str], trading_date: date) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    connection = conn()
    try:
        placeholders = ",".join("?" for _ in symbols)
        query = f"""
        SELECT
            symbol,
            SUM(((high + low + close) / 3.0) * volume) / NULLIF(SUM(volume), 0) AS vwap
        FROM candles
        WHERE interval = '1m'
          AND symbol IN ({placeholders})
          AND CAST(timestamp AT TIME ZONE 'Asia/Kolkata' AS DATE) = ?
        GROUP BY symbol
        """
        return connection.execute(query, [*symbols, trading_date]).df()
    finally:
        connection.close()


def _instrument_rows() -> pd.DataFrame:
    connection = conn()
    try:
        return connection.execute(
            "SELECT symbol, instrument_key FROM instrument_master ORDER BY symbol"
        ).df()
    finally:
        connection.close()


def scan_top10(
    client: UpstoxREST,
    config: ScannerConfig | None = None,
    trading_date: date | None = None,
) -> list[dict]:
    """Build the current NSE candidate list from stored history plus live quotes.

    The scanner scores every eligible symbol, persists the full ranked research
    universe, and returns only the configured top-N shortlist downstream.
    """
    config = config or ScannerConfig()
    captured_at = datetime.now(IST)
    trading_date = trading_date or captured_at.date()
    instruments = _instrument_rows()
    if instruments.empty:
        return []

    keys = instruments["instrument_key"].dropna().astype(str).tolist()
    quotes = client.full_market_quotes(keys)
    metrics = client.quote_metrics(quotes)
    quote_by_instrument: dict[str, dict] = {}
    metric_by_instrument: dict[str, dict] = {}
    for quote_key, value in quotes.items():
        if not isinstance(value, dict):
            continue
        instrument_key = value.get("instrument_token") or quote_key
        quote_by_instrument[str(instrument_key)] = value
        metric_by_instrument[str(instrument_key)] = metrics.get(quote_key, {})

    symbols = instruments["symbol"].tolist()
    liquidity = _historical_liquidity(symbols, trading_date, config.lookback_days)
    vwap = _current_day_vwap(symbols, trading_date)
    news = latest_symbol_news_scores(config.news_lookback_hours)
    liquidity_map = liquidity.set_index("symbol").to_dict("index") if not liquidity.empty else {}
    vwap_map = vwap.set_index("symbol")["vwap"].to_dict() if not vwap.empty else {}
    news_map = news.set_index("symbol").to_dict("index") if not news.empty else {}
    market_score = latest_market_score()

    rows: list[dict] = []
    elapsed = _elapsed_minutes(captured_at)
    for _, instrument in instruments.iterrows():
        symbol = str(instrument["symbol"])
        key = str(instrument["instrument_key"])
        raw = quote_by_instrument.get(key)
        metric = metric_by_instrument.get(key, {})
        if not raw or not metric.get("ltp"):
            continue

        ltp = float(metric["ltp"])
        if ltp < config.minimum_price:
            continue
        history = liquidity_map.get(symbol, {})
        avg_volume = float(history.get("avg_daily_volume") or 0.0)
        avg_value = float(history.get("avg_daily_traded_value") or 0.0)
        if avg_value < config.minimum_avg_daily_traded_value:
            continue

        news_row = news_map.get(symbol, {})
        ohlc = raw.get("ohlc") or {}
        rows.append(
            {
                "symbol": symbol,
                "instrument_key": key,
                "ltp": ltp,
                "previous_close": metric.get("previous_close"),
                "change_pct": metric.get("change_pct"),
                "cumulative_volume": float(raw.get("volume") or 0.0),
                "avg_daily_volume": avg_volume,
                "avg_daily_traded_value": avg_value,
                "day_high": float(ohlc.get("high") or ltp),
                "day_low": float(ohlc.get("low") or ltp),
                "vwap": vwap_map.get(symbol),
                "news_score": float(news_row.get("news_score") or 0.0),
                "news_count": int(news_row.get("news_count") or 0),
                "high_impact_news_count": int(news_row.get("high_impact_news_count") or 0),
                "elapsed_session_minutes": elapsed,
                "history_days": int(history.get("history_days") or 0),
            }
        )

    # Rank the complete eligible universe first. The previous implementation
    # truncated here, which meant candidate_events only retained top-N rows.
    ranked = rank_candidates(rows, market_score=market_score, limit=None)
    if not ranked:
        return []

    candidate_df = pd.DataFrame(
        [
            {
                "event_time": captured_at,
                "trading_date": trading_date,
                "symbol": row["symbol"],
                "instrument_key": row["instrument_key"],
                "ltp": row["ltp"],
                "volume": row["cumulative_volume"],
                "relative_volume": row["relative_volume"],
                "price_change_pct": row["change_pct"],
                "vwap": row.get("vwap"),
                "candidate_score": row["candidate_score"],
                "reason": row["reason"],
                "news_score": row.get("news_score", 0.0),
                "news_count": row.get("news_count", 0),
                "high_impact_news_count": row.get("high_impact_news_count", 0),
            }
            for row in ranked
        ]
    )
    insert_df("candidate_events", candidate_df)

    # Preserve the existing API contract: downstream paper trading receives
    # only the configured shortlist, while the database retains every row.
    return ranked[:config.limit]
