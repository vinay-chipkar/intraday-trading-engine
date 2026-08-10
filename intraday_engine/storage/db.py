from __future__ import annotations

import duckdb
import pandas as pd

from config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS market_context(
    captured_at TIMESTAMPTZ,
    trading_date DATE,
    gift_nifty_change_pct DOUBLE,
    dow_change_pct DOUBLE,
    sp500_change_pct DOUBLE,
    nasdaq_change_pct DOUBLE,
    india_vix DOUBLE,
    usd_inr DOUBLE,
    brent DOUBLE,
    fii_flow DOUBLE,
    dii_flow DOUBLE,
    nifty_change_pct DOUBLE,
    banknifty_change_pct DOUBLE,
    news_count INTEGER,
    high_impact_news_count INTEGER,
    score DOUBLE,
    regime VARCHAR
);

CREATE TABLE IF NOT EXISTS instrument_master(
    symbol VARCHAR PRIMARY KEY,
    instrument_key VARCHAR,
    name VARCHAR,
    trading_symbol VARCHAR,
    updated_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS candles(
    instrument_key VARCHAR,
    symbol VARCHAR,
    timestamp TIMESTAMPTZ,
    interval VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    open_interest DOUBLE,
    PRIMARY KEY (instrument_key, timestamp, interval)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(timestamp);

CREATE TABLE IF NOT EXISTS ingestion_runs(
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    mode VARCHAR,
    interval VARCHAR,
    requested_symbols INTEGER,
    successful_symbols INTEGER,
    rows_received BIGINT,
    rows_inserted BIGINT,
    status VARCHAR,
    error VARCHAR
);

CREATE TABLE IF NOT EXISTS data_quality_events(
    event_time TIMESTAMPTZ,
    symbol VARCHAR,
    instrument_key VARCHAR,
    timestamp TIMESTAMPTZ,
    issue_type VARCHAR,
    details VARCHAR
);

CREATE TABLE IF NOT EXISTS candidate_events(
    event_time TIMESTAMPTZ,
    trading_date DATE,
    symbol VARCHAR,
    instrument_key VARCHAR,
    ltp DOUBLE,
    volume DOUBLE,
    relative_volume DOUBLE,
    price_change_pct DOUBLE,
    vwap DOUBLE,
    candidate_score DOUBLE,
    reason VARCHAR
);

CREATE TABLE IF NOT EXISTS feature_snapshots(
    event_time TIMESTAMPTZ,
    trading_date DATE,
    symbol VARCHAR,
    instrument_key VARCHAR,
    timeframe VARCHAR,
    close DOUBLE,
    volume DOUBLE,
    relative_volume DOUBLE,
    vwap DOUBLE,
    rsi14 DOUBLE,
    ema9 DOUBLE,
    ema20 DOUBLE,
    ema50 DOUBLE,
    ema200 DOUBLE,
    atr14 DOUBLE,
    support DOUBLE,
    resistance DOUBLE,
    distance_to_support_pct DOUBLE,
    distance_to_resistance_pct DOUBLE,
    candle_pattern VARCHAR,
    trend VARCHAR,
    breakout BOOLEAN,
    breakdown BOOLEAN,
    feature_score DOUBLE,
    feature_json VARCHAR
);

CREATE TABLE IF NOT EXISTS signals(
    signal_id BIGINT,
    event_time TIMESTAMPTZ,
    trading_date DATE,
    symbol VARCHAR,
    side VARCHAR,
    entry_low DOUBLE,
    entry_high DOUBLE,
    stop_loss DOUBLE,
    target1 DOUBLE,
    target2 DOUBLE,
    risk_reward DOUBLE,
    score DOUBLE,
    confidence DOUBLE,
    setup VARCHAR,
    reasons VARCHAR,
    status VARCHAR
);

CREATE TABLE IF NOT EXISTS training_labels(
    event_time TIMESTAMPTZ,
    symbol VARCHAR,
    horizon_minutes INTEGER,
    entry_price DOUBLE,
    target_price DOUBLE,
    stop_price DOUBLE,
    target_hit_first BOOLEAN,
    max_favorable_excursion DOUBLE,
    max_adverse_excursion DOUBLE,
    label INTEGER
);
"""


CANDLE_COLUMNS = [
    "instrument_key", "symbol", "timestamp", "interval",
    "open", "high", "low", "close", "volume", "open_interest",
]

MARKET_CONTEXT_COLUMNS = [
    "captured_at", "trading_date", "gift_nifty_change_pct", "dow_change_pct",
    "sp500_change_pct", "nasdaq_change_pct", "india_vix", "usd_inr", "brent",
    "fii_flow", "dii_flow", "nifty_change_pct", "banknifty_change_pct",
    "news_count", "high_impact_news_count", "score", "regime",
]

FEATURE_SNAPSHOT_COLUMNS = [
    "event_time", "trading_date", "symbol", "instrument_key", "timeframe",
    "close", "volume", "relative_volume", "vwap", "rsi14", "ema9", "ema20",
    "ema50", "ema200", "atr14", "support", "resistance", "distance_to_support_pct",
    "distance_to_resistance_pct", "candle_pattern", "trend", "breakout", "breakdown",
    "feature_score", "feature_json",
]


def conn():
    connection = duckdb.connect(settings.duckdb_path)
    connection.execute(SCHEMA)
    return connection


def insert_df(table: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    connection = conn()
    try:
        connection.register("incoming_df", df)
        connection.execute(f"INSERT INTO {table} SELECT * FROM incoming_df")
    finally:
        connection.unregister("incoming_df")
        connection.close()


def upsert_instruments(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    connection = conn()
    try:
        connection.register("incoming_df", df)
        connection.execute("INSERT OR REPLACE INTO instrument_master SELECT * FROM incoming_df")
    finally:
        connection.unregister("incoming_df")
        connection.close()


def insert_market_context(values: dict) -> None:
    missing = set(MARKET_CONTEXT_COLUMNS).difference(values)
    if missing:
        raise ValueError(f"Missing market context columns: {sorted(missing)}")
    connection = conn()
    try:
        connection.execute(
            "INSERT INTO market_context VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [values[column] for column in MARKET_CONTEXT_COLUMNS],
        )
    finally:
        connection.close()


def insert_feature_snapshot(values: dict) -> None:
    """Persist one point-in-time feature vector for later research/ML labels."""
    missing = set(FEATURE_SNAPSHOT_COLUMNS).difference(values)
    if missing:
        raise ValueError(f"Missing feature snapshot columns: {sorted(missing)}")
    connection = conn()
    try:
        placeholders = ",".join("?" for _ in FEATURE_SNAPSHOT_COLUMNS)
        connection.execute(
            f"INSERT INTO feature_snapshots VALUES ({placeholders})",
            [values[column] for column in FEATURE_SNAPSHOT_COLUMNS],
        )
    finally:
        connection.close()


def insert_candles(df: pd.DataFrame) -> int:
    """Insert candles idempotently and return the number of new rows."""
    if df is None or df.empty:
        return 0

    missing = set(CANDLE_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    connection = conn()
    try:
        ordered = df[CANDLE_COLUMNS].copy()
        connection.register("incoming_df", ordered)
        before = connection.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        connection.execute("INSERT OR IGNORE INTO candles SELECT * FROM incoming_df")
        after = connection.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        return int(after - before)
    finally:
        connection.unregister("incoming_df")
        connection.close()


def get_instruments(symbols: list[str] | None = None) -> pd.DataFrame:
    connection = conn()
    try:
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            return connection.execute(
                f"SELECT symbol, instrument_key, name, trading_symbol FROM instrument_master WHERE symbol IN ({placeholders}) ORDER BY symbol",
                [s.upper() for s in symbols],
            ).df()
        return connection.execute(
            "SELECT symbol, instrument_key, name, trading_symbol FROM instrument_master ORDER BY symbol"
        ).df()
    finally:
        connection.close()


def latest_candle_timestamp(instrument_key: str, interval: str = "1m"):
    connection = conn()
    try:
        row = connection.execute(
            "SELECT MAX(timestamp) FROM candles WHERE instrument_key = ? AND interval = ?",
            [instrument_key, interval],
        ).fetchone()
        return row[0] if row else None
    finally:
        connection.close()


def latest_market_context() -> dict | None:
    connection = conn()
    try:
        row = connection.execute(
            "SELECT * FROM market_context ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return dict(zip(MARKET_CONTEXT_COLUMNS, row))
    finally:
        connection.close()


def latest_market_score():
    context = latest_market_context()
    return float(context["score"]) if context else 0.0
