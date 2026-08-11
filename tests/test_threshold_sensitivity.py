from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_engine.backtest.engine import BacktestTrade
from intraday_engine.research.threshold_sensitivity import _metrics, _prepare_symbol, run_threshold_sweep
from intraday_engine.strategy.point_in_time import enrich_point_in_time


def _trade(pnl_points: float, r_multiple: float, exit_time: str, symbol: str = "TEST", outcome: str = "TARGET") -> BacktestTrade:
    return BacktestTrade(
        symbol=symbol,
        side="BUY",
        signal_time=pd.Timestamp(exit_time),
        entry_time=pd.Timestamp(exit_time),
        exit_time=pd.Timestamp(exit_time),
        entry_price=100.0,
        exit_price=100.0 + pnl_points,
        stop_loss=99.0,
        target=101.0,
        outcome=outcome,
        pnl_points=pnl_points,
        r_multiple=r_multiple,
        holding_bars=1,
    )


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


def test_metrics_profit_factor_uses_true_gross_profit_and_loss():
    # Net across all trades is positive (+25), which a net-points-based
    # approximation would read as "no losing trades" (profit_factor=None).
    # The true gross profit/loss split (30 / 5) gives a very different answer.
    trades = [
        _trade(10.0, 1.0, "2026-01-05 09:20"),
        _trade(10.0, 1.0, "2026-01-05 09:21"),
        _trade(10.0, 1.0, "2026-01-05 09:22"),
        _trade(-5.0, -1.0, "2026-01-05 09:23"),
    ]
    metrics = _metrics(trades)
    assert metrics["profit_factor"] == 6.0


def test_metrics_profit_factor_is_none_only_when_there_are_no_losses():
    trades = [_trade(10.0, 1.0, "2026-01-05 09:20"), _trade(5.0, 0.5, "2026-01-05 09:21")]
    metrics = _metrics(trades)
    assert metrics["profit_factor"] is None


def test_metrics_max_drawdown_reflects_combined_equity_curve():
    # Peak after trade 1 is +10; trade 2 drops equity to -5, a 15-point
    # drawdown from that peak; trade 3 partially recovers but does not
    # exceed the earlier peak.
    trades = [
        _trade(10.0, 1.0, "2026-01-05 09:20"),
        _trade(-15.0, -1.5, "2026-01-05 09:21", outcome="STOP"),
        _trade(5.0, 0.5, "2026-01-05 09:22"),
    ]
    metrics = _metrics(trades)
    assert metrics["max_drawdown_points"] == 15.0


def test_run_threshold_sweep_reports_nondegenerate_profit_factor_and_drawdown():
    # A choppy, mean-reverting price series (alternating drift per day) with
    # a fixed seed reliably produces both winning and losing trades, so a
    # regression back to the old net-points-based aggregation (which could
    # only ever yield profit_factor of None/0.0 and max_drawdown of 0.0)
    # would be caught here.
    rng = np.random.default_rng(1)
    rows = []
    price = 100.0
    for day in range(6):
        start = pd.Timestamp("2026-01-05") + pd.Timedelta(days=day)
        drift = 0.05 if day % 2 == 0 else -0.05
        for i in range(80):
            price += drift + rng.normal(0, 0.15)
            ts = start + pd.Timedelta(minutes=9 * 60 + 15 + i)
            rows.append({
                "timestamp": ts, "symbol": "TEST",
                "open": price - 0.05, "high": price + 0.15, "low": price - 0.15,
                "close": price, "volume": 1000 + i * 2,
            })
    bars = pd.DataFrame(rows)

    result = run_threshold_sweep(bars, thresholds=(40,), max_holding_bars=10)
    row = result.iloc[0]
    assert row["train_wins"] > 0 and row["train_losses"] > 0
    assert row["train_profit_factor"] not in (None, 0.0)
    assert row["train_max_drawdown_points"] > 0.0


def test_run_threshold_sweep_handles_tz_aware_candles():
    # Real Upstox-sourced candles are tz-aware (e.g. Etc/UTC); split_date is
    # derived from .dt.date, which is always tz-naive. Comparing the two
    # directly raises "Cannot compare tz-naive and tz-aware timestamps" -
    # this must work identically regardless of the source data's tz-awareness.
    naive = _bars()
    aware = naive.copy()
    aware["timestamp"] = aware["timestamp"].dt.tz_localize("UTC")

    result_naive = run_threshold_sweep(naive, thresholds=(60,), max_holding_bars=5)
    result_aware = run_threshold_sweep(aware, thresholds=(60,), max_holding_bars=5)
    cols = ["train_trades", "train_win_rate_pct", "train_profit_factor", "train_expectancy_r",
            "oos_trades", "oos_win_rate_pct", "oos_profit_factor", "oos_expectancy_r"]
    assert result_naive[cols].to_dict("records") == result_aware[cols].to_dict("records")


def test_prepare_symbol_structure_trend_matches_live_point_in_time_pivot_trend():
    # Diagnostics found "market structure" and "EMA trend" producing identical
    # numbers for every cut because _prepare_symbol set structure_trend = trend
    # (a literal copy of the EMA-order trend) instead of the causal pivot-based
    # structure that the live pipeline (strategy/point_in_time.py) actually uses.
    # structure_trend must now come from the same function as the live "trend"
    # column, not a reimplementation of it.
    bars = _bars(days=8, bars_per_day=40)
    prepared = _prepare_symbol(bars)
    live = enrich_point_in_time(bars)
    assert prepared["structure_trend"].tolist() == live["trend"].tolist()
    # and it must be a genuinely independent signal from the EMA-order trend,
    # not a duplicate of it -- some bars have to disagree for this dataset.
    assert (prepared["structure_trend"] != prepared["trend"]).any()


def test_structure_trend_does_not_use_future_bars():
    bars = _bars(days=8, bars_per_day=40)
    mutated = bars.copy()
    cutoff = len(mutated) - 20
    mutated.loc[mutated.index[cutoff:], ["open", "high", "low", "close"]] += 1000

    a = _prepare_symbol(bars)
    b = _prepare_symbol(mutated)
    assert a["structure_trend"].iloc[:cutoff].tolist() == b["structure_trend"].iloc[:cutoff].tolist()
