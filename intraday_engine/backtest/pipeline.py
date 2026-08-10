from __future__ import annotations

import pandas as pd

from intraday_engine.backtest.engine import BacktestResult, backtest_signals
from intraday_engine.strategy.point_in_time import generate_signals


def run_rule_backtest(
    bars: pd.DataFrame,
    *,
    symbol: str,
    market_score: float = 0.0,
    min_score: float = 60.0,
    max_holding_bars: int = 30,
    slippage_points: float = 0.0,
) -> BacktestResult:
    """Generate causal rule signals and evaluate them on the same OHLCV series.

    Feature generation is point-in-time. The backtest engine then enforces
    next-bar execution, so future bars are only used for outcome evaluation.
    """
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ordered = bars.sort_values("timestamp").reset_index(drop=True).copy()
    ordered["symbol"] = symbol
    signals = generate_signals(
        ordered,
        symbol=symbol,
        market_score=market_score,
        min_score=min_score,
    )
    return backtest_signals(
        signals,
        ordered,
        max_holding_bars=max_holding_bars,
        slippage_points=slippage_points,
    )
