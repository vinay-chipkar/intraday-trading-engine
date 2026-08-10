# Research and Paper-Trading Protocol

The engine is research-first. No live order-placement path is part of this protocol.

## Premarket

Run the market snapshot before the cash session:

```bash
python -m scripts.premarket
```

The snapshot combines global/index quote context, India VIX, and recent Upstox instrument news. News is stored in `market_news` and contributes a bounded component to the market regime score. Upstox limits news requests to 30 instrument keys, so the implementation batches the instrument universe.

Then rank the current universe:

```bash
python -m scripts.scan --limit 10
```

The scanner ranks liquid instruments using market alignment, time-adjusted RVOL, momentum, intraday range, and recent symbol-specific news. It does not create a trade by itself.

## Research loop

For each research period:

1. Backfill clean 1-minute candles.
2. Run the normal backtest.
3. Run `scripts.stress_backtest` across multiple score thresholds and slippage assumptions.
4. Run `scripts.research_filters` for declared filters.
5. Prefer chronological out-of-sample results over in-sample winners.
6. Store signal outcomes, MFE/MAE, market regime, news context, and rejection reasons.
7. Change one strategy component at a time and re-run the full regression suite.

Example:

```bash
python -m scripts.stress_backtest /tmp/backtest_data
python -m scripts.research_filters /tmp/backtest_data
pytest
```

## Paper risk guardrails

The in-memory `PaperBroker` is deliberately isolated from broker APIs and enforces:

- risk-based position sizing
- maximum position count
- maximum position notional
- configurable slippage
- daily loss halt
- automatic reset of the daily loss guard on the next trading day
- forced end-of-day close support

Default paper settings are conservative:

- initial capital: 100,000
- risk per trade: 0.5%
- maximum open positions: 3
- maximum daily loss: 2%
- maximum position notional: 25% of capital
- simulated slippage: 0.10 points

These are simulation controls, not recommendations for live trading.

## Six-month rule

Do not add a live order-placement integration merely because paper results look good.

Before considering live execution, require:

- multiple months of untouched out-of-sample paper results
- positive expectancy after realistic slippage/cost assumptions
- stable performance across multiple symbols and market regimes
- no unexplained data-quality failures
- documented behavior during news shocks and market gaps
- a separately tested live-execution layer with explicit kill-switch controls
