from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_engine.research.threshold_sensitivity import run_threshold_sweep


def _bars(days: int = 4, bars_per_day: int = 80) -> pd.DataFrame:
    rows = []
    for day in range(days):
        start = pd.Timestamp("2026-01-05") + pd.Timedelta(days=day)
        for i in range(bars_per_day):
            ts = start + pd.Timedelta(minutes=9 * 60 + 15 + i)
            close = 100 + day * 2 + i * 0.03 + np.sin(i / 7) * 0.2
            rows.append({
                "timestamp": ts,
                "symbol": "TEST",
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": 1000 + i * 2,
            })
    return pd.DataFrame(rows)


def test_threshold_sweep_returns_all_requested_thresholds():
    result = run_threshold_sweep(_bars(), thresholds=(40, 60), max_holding_bars=5)
    assert result["threshold"].tolist() == [40.0, 60.0]
    assert all(result["train_trades"] >= 0)
    assert all(result["oos_trades"] >= 0)


def test_future_bar_mutation_does_not_change_train_metrics():
    original = _bars()
    mutated = original.copy()
    split = mutated["timestamp"].dt.date.unique()[3]
    mask = mutated["timestamp"].dt.date == split
    mutated.loc[mask, ["open", "high", "low", "close"]] += 1000

    a = run_threshold_sweep(original, thresholds=(60,), max_holding_bars=5)
    b = run_threshold_sweep(mutated, thresholds=(60,), max_holding_bars=5)
    cols = ["train_trades", "train_win_rate_pct", "train_profit_factor", "train_expectancy_r"]
    assert a[cols].to_dict("records") == b[cols].to_dict("records")
