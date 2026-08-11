# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research-first intraday trading engine for NSE equities using Upstox market data. It deliberately separates market data, feature generation, pattern detection, scoring, paper trading, backtesting labels, and future ML/execution. **No live order placement is implemented** — do not add it until there is a statistically meaningful backtest, out-of-sample validation, realistic transaction-cost/slippage modelling, and an extended paper-trading period.

## Commands

```bash
pip install -e .          # install (editable), also required after any dependency change
pytest                    # run all tests
pytest -q tests/test_signal_engine.py            # run one test file
pytest tests/test_signal_engine.py::test_name     # run one test
```

No linter/formatter is configured (no ruff/black/mypy/flake8 config, no Makefile). `pythonpath = ["."]` is set in `pyproject.toml` so tests import `intraday_engine`/`config` directly from the repo root without installing.

### Entry points (`scripts/`, run as `python -m scripts.<name>` or `python scripts/<name>.py`)

- `sync_instruments.py` — resolve configured symbols to Upstox instrument keys (run first, before anything else).
- `premarket.py` — capture the macro/regime snapshot into `market_context`.
- `daily_cycle.py` — orchestrates sync → premarket → ingestion → `scan_top10` (`intraday_engine/research/daily_cycle.py::run_daily_cycle`); paper-mode only.
- `scan.py` — run the intraday universe scanner standalone.
- `paper_session.py --interval 5 --until 12:00 [--no-bootstrap]` — forward paper-trading loop for a session window.
- `paper_close.py` — finalize a paper session, evaluate outcomes, export the research journal.
- `restore_paper_journal.py` — restore a previously exported paper journal (used to seed CI runs).
- `backtest.py data.csv --symbol X` — deterministic rule backtest against a CSV of `timestamp,open,high,low,close[,volume]`.
- `backfill.py` / `backfill_history.py` — historical candle backfill.
- `threshold_sweep.py` / `threshold_sweep_fast.py` — sweep signal score thresholds for train/test sensitivity.

The `paper-research.yml` GitHub Actions workflow runs the real daily cycle (morning bootstrap + session, afternoon session + close, journal commit) on a 09:00 IST weekday cron — this is the closest thing to a "production" run of this repo.

### Setup

```bash
cp .env.example .env   # then set UPSTOX_ACCESS_TOKEN
```

DuckDB file path, timezone, paper-trading risk params, and Upstox global-instrument keys are all read from `.env` via `config/settings.py::settings` — see `.env.example` for the full list. `config/universe.csv` is the starter liquid NSE equity universe.

## Architecture

### DuckDB is the backbone (`intraday_engine/storage/db.py`)

Every stage reads/writes DuckDB tables rather than passing data directly between stages: `market_context`, `market_news`, `instrument_master`, `candles`, `ingestion_runs`, `data_quality_events`, `candidate_events` (scanner output), `feature_snapshots` (point-in-time feature vectors for research/ML), `signals`, `training_labels`, `research_runs`, plus per-module tables owned by `research/*` (`ensure_*_table`). There is **no persistent connection or pool** — `conn()` opens a fresh DuckDB connection per call, runs schema DDL/migrations, and every function `close()`s it in a `finally`. Follow this pattern for new DB code; don't hold a long-lived handle. Column-list constants (`CANDLE_COLUMNS`, `MARKET_CONTEXT_COLUMNS`, `FEATURE_SNAPSHOT_COLUMNS`, `NEWS_COLUMNS`) are contracts — DataFrames passed to `insert_df`/bulk inserts must match column order exactly.

### Upstox is the only network boundary (`intraday_engine/market/upstox.py`)

`UpstoxREST` is the single class every other market module depends on for live data: instrument search/resolution (with `SYMBOL_ALIASES` for renamed tickers), intraday/historical candles (normalized to OHLCV DataFrames), full market quotes/metrics, and news. `market/universe.py`, `market/context.py`, `market/news.py`, `market/ingestion.py`, `market/backfill.py` all build on it; `market/candles.py` is pure normalization/quality-check logic with no network calls.

### There are three separate point-in-time enrichment implementations — know which one is live

This repo's core data-integrity rule (from the README): every decision point must only use features available **at that timestamp**; the backtest executes on the *next* bar after a signal, never the signal bar itself, so the six-month dataset stays free of look-ahead leakage when used for ML.

- `intraday_engine/technical/structure.py::swing_points` uses a **centered rolling window** — it looks at future bars and is *not* point-in-time by itself.
- `intraday_engine/strategy/features.py::enrich()` is a legacy path that calls the centered-window swing/support-resistance/trend functions directly — it leaks look-ahead bias and is not used by the live pipeline.
- `intraday_engine/strategy/point_in_time.py::enrich_point_in_time()` is the **causal, pipeline-live** replacement: it recomputes pivots itself (`_confirmed_pivots`), shifts them so a pivot is only "known" `right` bars after it forms, then walks forward building support/resistance/trend from only pivots confirmed so far. `generate_signals()` in this same module is the glue that runs `enrich_point_in_time` and then calls `signals/engine.py::generate_signal` row-by-row — this is what `backtest/pipeline.py`, `backtest/research.py`, and the research/paper modules actually call.
- `intraday_engine/technical/feature_engine.py::add_feature_engine`/`latest_feature_snapshot` is a **third, independent** point-in-time implementation used specifically to populate `feature_snapshots` for storage/ML — it does not share code with `strategy/point_in_time.py`.

When touching signal/feature logic, default to `strategy/point_in_time.py` as the live path unless you're specifically working on the `feature_snapshots`/ML side.

### Scanner vs. signal engine — different jobs, not wired together

- `scanner/service.py::scan_universe`/`scan_top10` decides **which symbols to watch**: live quotes + historical liquidity + VWAP + news, scored by `scanner/ranking.py::rank_candidates`, persisted to `candidate_events`.
- `signals/engine.py::generate_signal` decides **whether/how to trade a given bar**: trend/structure/EMA/VWAP/RSI/MACD/ADX/RVOL/ORB/candlestick-pattern score with blockers (weak ADX, weak RVOL, extended-from-VWAP), emitting a `TradeSignal` (BUY/SELL/NO_TRADE) with a stop/target.
- These two are independent stages today — `research/daily_cycle.py::run_daily_cycle` stops at scanner output (candidates); it does not currently feed candidates into signal generation.

### Backtest vs. paper trading — two independent trade simulators

- `backtest/engine.py::backtest_signals` is the deterministic, vectorized historical simulator: given pre-generated `TradeSignal`s and an OHLCV frame, executes at the next bar's open and walks forward checking stop/target/timeout. `backtest/pipeline.py`/`backtest/research.py` build whole-history research runs on top of it (plus `strategy.point_in_time.generate_signals`). `backtest/diagnostics.py` adds trade-level diagnostics.
- `paper/simulator.py::PaperBroker` is a stateful, incremental broker (risk-based position sizing, max positions, daily-loss halt) driven bar-by-bar during real/replayed sessions by `scripts/paper_session.py`.
- They do **not** share stop/target logic — if you change exit rules, you likely need to update both.

### `intraday_engine/research/` is the orchestration layer above `scripts/`

`scripts/*.py` are thin CLI wrappers; the actual logic lives in `research/`: `daily_cycle.py` (sync → premarket → ingestion → scan), `paper_observer.py` (persist point-in-time paper-trade observations from live signals), `paper_outcomes.py` (evaluate pending observations against subsequent bars into outcome labels), `paper_learning.py` (aggregate outcome patterns into a learning report, without touching strategy params), `threshold_sensitivity.py` (score-threshold sweeps with train/test splits).

### `intraday_engine/ml/` is a standalone scaffold, not wired in

`ml/labels.py` (pure forward-labeling functions) and `ml/train.py` (`HistGradientBoostingClassifier` over the `feature_snapshots` column set) are not imported by any script or research module yet — the `training_labels` table exists but nothing currently populates/consumes it end-to-end. The intended ML target is the probability a defined setup reaches its target before its stop within a fixed horizon, not a raw BUY/SELL prediction, to be evaluated alongside rule-based signals rather than replacing them.

## Project status

Done: code foundation, backtesting, causality safeguards, stress testing, research framework.

In progress: dynamic universe selection, a 9 AM market/news engine, walk-forward validation, realistic transaction costs, the paper-trading engine, automatic mistake analysis/model retraining.

## Working philosophy

This system will eventually risk real capital, so correctness over speed on every change:

- A previous "fast" rewrite of the threshold-sweep code already introduced a silent regression once — treat performance rewrites of research/backtest code as high-risk, not routine.
- Explain the root cause of a bug before changing code. Don't jump straight to a fix.
- Verify fixes against actual output (print/compare real values), not just "tests pass."
- Flag anything else you notice along the way, even if it wasn't asked about.
