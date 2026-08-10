# Intraday Trading Engine

Research-first intraday engine for NSE equities using Upstox. The design deliberately separates market data, feature generation, pattern detection, scoring, paper trading, backtesting labels, and future ML/execution.

## Current pipeline

```text
09:00 market context
        -> instrument/universe sync
        -> historical/intraday candle ingestion
        -> DuckDB
        -> technical features
        -> candlestick + market structure
        -> candidate scoring / top 10
        -> signal generation
        -> paper trading
        -> outcome labels
        -> six-month dataset
        -> ML probability model
        -> future live execution adapter
```

**No live order placement is implemented.** The execution layer is intentionally left out until the research and paper-trading results are validated.

## Repository layout

- `intraday_engine/market/` — Upstox REST adapter, market context, instrument resolution, candle normalization, historical backfill, and incremental ingestion.
- `intraday_engine/technical/` — EMA, RSI, ATR, VWAP, volume, Bollinger Bands, swing structure, breakout/breakdown.
- `intraday_engine/patterns/` — candlesticks and double-top/double-bottom detection.
- `intraday_engine/strategy/` — feature enrichment, scoring and signal construction.
- `intraday_engine/paper/` — risk-based position sizing and paper broker.
- `intraday_engine/ml/` — forward labels and ML training scaffold.
- `intraday_engine/backtest/` — deterministic signal simulation and performance metrics.
- `intraday_engine/storage/` — DuckDB schema and persistence.
- `config/universe.csv` — starter liquid NSE equity universe.
- `scripts/` — runnable entry points.
- `tests/` — unit tests.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `UPSTOX_ACCESS_TOKEN` in `.env`.

## Real-data pipeline

### 1. Resolve instruments

```bash
python scripts/sync_instruments.py
```

This stores the current Upstox `instrument_key` for each symbol in `instrument_master`.

### 2. Backfill historical 1-minute candles

```bash
python scripts/backfill.py --symbol RELIANCE --days 30
```

Multiple symbols can be supplied:

```bash
python scripts/backfill.py --symbol RELIANCE --symbol TCS --symbol INFY --days 30
```

Or use the configured universe:

```bash
python scripts/backfill.py --universe --days 30
```

The service splits minute-data requests into 30-calendar-day windows because Upstox V3 limits minute intervals to roughly one month per historical request. citeturn0search1turn0search8

### 3. Incrementally ingest the current trading day

```bash
python scripts/ingest.py
```

Or for selected stocks:

```bash
python scripts/ingest.py --symbol RELIANCE --symbol TCS
```

The current-day Upstox intraday endpoint is fetched, but only candles newer than the latest persisted timestamp are inserted. The database primary key also makes the operation idempotent. Upstox V3 supports configurable intraday minute intervals. citeturn0search4

### 4. Inspect DuckDB

The database is created at the path configured by `DUCKDB_PATH` and contains at least:

```text
instrument_master
candles
ingestion_runs
market_context
candidate_events
feature_snapshots
signals
training_labels
```

The candle key is:

```text
instrument_key + timestamp + interval
```

This prevents duplicate candles when ingestion runs repeatedly.

## Upstox data sources

The engine uses Upstox's instrument search API for symbol-to-instrument resolution. The API supports NSE/EQ filtering and returns the `instrument_key` needed by the market-data APIs. citeturn0search3turn0search0

Historical minute candles use Upstox Historical Candle V3. Current-day candles use Intraday Candle V3. citeturn0search1turn0search4

Upstox also exposes global instruments such as GIFT NIFTY, major global indices, USD/INR and oil indicators through its market-data APIs; those will be wired into the pre-market context in the next milestone. citeturn0search11

## Data philosophy

Every decision point must preserve the features available **at that timestamp**. Future candles are only used later to create outcome labels. This prevents look-ahead leakage when the six-month dataset is eventually used for ML.

The intended ML target is not a raw BUY/SELL prediction. The first model should estimate the probability that a defined setup reaches its target before its stop within a fixed horizon. Rule-based signals and ML probability can then be evaluated together.

## Research boundary

This repository is a research and paper-trading system. It does not place live orders. Do not enable live execution until there is a statistically meaningful backtest, out-of-sample validation, realistic transaction-cost/slippage modelling, and an extended paper-trading period.
