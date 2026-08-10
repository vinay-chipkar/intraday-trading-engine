# Intraday Trading Engine

Research-first intraday engine for NSE equities using Upstox. The design deliberately separates market data, feature generation, pattern detection, scoring, paper trading, backtesting labels, and future ML/execution.

## Current pipeline

```text
09:00 market context
        -> instrument/universe sync
        -> intraday candle ingestion
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

- `intraday_engine/market/` — Upstox REST adapter, market context, instrument resolution.
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

## First run

1. Resolve the current Upstox instrument keys:

```bash
python scripts/sync_instruments.py
```

2. Capture pre-market context. `scripts/premarket.py` pulls global/index data through Upstox and leaves FII/DII/news as explicit provider inputs.

```bash
python scripts/premarket.py
```

3. Run the scanner during market hours:

```bash
python scripts/scan.py
```

4. Run tests:

```bash
pytest
```

## Data philosophy

Every decision point should preserve the features available **at that timestamp**. Future candles are only used later to create outcome labels. This prevents look-ahead leakage when the six-month dataset is eventually used for ML.

The intended ML target is not a raw BUY/SELL prediction. The first model should estimate the probability that a defined setup reaches its target before its stop within a fixed horizon. Rule-based signals and ML probability can then be evaluated together.

## Safety / research boundary

This repository is a research and paper-trading system. It does not place live orders. Do not enable live execution until there is a statistically meaningful backtest, out-of-sample validation, realistic transaction-cost/slippage modelling, and an extended paper-trading period.

## Upstox notes

The adapter uses Upstox V3 intraday/historical candle APIs and the instrument-search API. Upstox recommends using `instrument_key` as the stable identifier rather than `exchange_token`.
