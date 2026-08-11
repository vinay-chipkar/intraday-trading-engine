from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_engine.research.diagnostics import (
    adx_bucket,
    build_diagnostic_trades,
    cluster_significance,
    entry_time_bucket,
    holding_bucket,
    rvol_bucket,
    score_bucket,
    summarize,
    vwap_distance_bucket,
    _component_support,
    _group_metrics,
)
from intraday_engine.research.threshold_sensitivity import run_threshold_sweep


def _bars(days: int = 6, bars_per_day: int = 80) -> pd.DataFrame:
    rows = []
    for day in range(days):
        start = pd.Timestamp("2026-01-05") + pd.Timedelta(days=day)
        for i in range(bars_per_day):
            ts = start + pd.Timedelta(minutes=9 * 60 + 15 + i)
            close = 100 + day * 2 + i * 0.03 + np.sin(i / 7) * 0.2
            rows.append({
                "timestamp": ts, "symbol": "TEST",
                "open": close - 0.05, "high": close + 0.10, "low": close - 0.10,
                "close": close, "volume": 1000 + i * 2,
            })
    return pd.DataFrame(rows)


def _all_bullish_row() -> pd.Series:
    return pd.Series({
        "close": 105.0, "ema9": 104.0, "ema20": 103.0, "ema50": 102.0, "vwap": 100.0,
        "rsi14": 60.0, "adx14": 30.0, "plus_di14": 25.0, "minus_di14": 15.0,
        "macd_histogram": 1.0, "relative_volume": 2.0,
        "trend": "UPTREND", "structure_trend": "UPTREND",
        "opening_range_breakout": True, "opening_range_breakdown": False,
        "hammer": True, "bullish_engulfing": False, "morning_star": False,
        "shooting_star": False, "bearish_engulfing": False, "evening_star": False,
    })


def _all_bearish_row() -> pd.Series:
    return pd.Series({
        "close": 95.0, "ema9": 96.0, "ema20": 97.0, "ema50": 98.0, "vwap": 100.0,
        "rsi14": 40.0, "adx14": 30.0, "plus_di14": 15.0, "minus_di14": 25.0,
        "macd_histogram": -1.0, "relative_volume": 2.0,
        "trend": "DOWNTREND", "structure_trend": "DOWNTREND",
        "opening_range_breakout": False, "opening_range_breakdown": True,
        "hammer": False, "bullish_engulfing": False, "morning_star": False,
        "shooting_star": True, "bearish_engulfing": False, "evening_star": False,
    })


def test_component_support_all_align_bullish():
    support = _component_support(_all_bullish_row(), bullish=True)
    assert all(support.values())


def test_component_support_all_align_bearish():
    support = _component_support(_all_bearish_row(), bullish=False)
    assert all(support.values())


def test_component_support_neutral_rsi_does_not_support_either_direction():
    row = _all_bullish_row()
    row["rsi14"] = 50.0  # inside neither the 52-70 nor 30-48 window
    assert _component_support(row, bullish=True)["rsi"] is False
    assert _component_support(row, bullish=False)["rsi"] is False


def test_component_support_opposite_direction_does_not_support():
    bearish_row = _all_bearish_row()
    assert _component_support(bearish_row, bullish=True)["ema_trend"] is False
    assert _component_support(bearish_row, bullish=True)["adx"] is False


def test_score_bucket_boundaries():
    assert score_bucket(40.0) == "40-49"
    assert score_bucket(49.9) == "40-49"
    assert score_bucket(50.0) == "50-59"
    assert score_bucket(69.9) == "60-69"
    assert score_bucket(70.0) == "70+"
    assert score_bucket(95.0) == "70+"


def test_entry_time_bucket_boundaries():
    assert entry_time_bucket(pd.Timestamp("2026-01-05 09:15")) == "09:15-10:00"
    assert entry_time_bucket(pd.Timestamp("2026-01-05 09:59")) == "09:15-10:00"
    assert entry_time_bucket(pd.Timestamp("2026-01-05 10:00")) == "10:00-11:00"
    assert entry_time_bucket(pd.Timestamp("2026-01-05 15:29")) == "14:00-15:30"
    assert entry_time_bucket(pd.Timestamp("2026-01-05 15:30")) == "OUTSIDE_SESSION"


def test_holding_bucket_boundaries():
    assert holding_bucket(1) == "1-5"
    assert holding_bucket(5) == "1-5"
    assert holding_bucket(6) == "6-10"
    assert holding_bucket(30) == "26-30"
    assert holding_bucket(31) == "31+"


def _trade_row(pnl: float, r: float, exit_time: str, symbol: str = "TEST") -> dict:
    return {"symbol": symbol, "pnl_points": pnl, "r_multiple": r, "exit_time": pd.Timestamp(exit_time)}


def test_group_metrics_profit_factor_and_win_rate():
    trades = pd.DataFrame([
        _trade_row(10.0, 1.0, "2026-01-05 09:20"),
        _trade_row(10.0, 1.0, "2026-01-05 09:21"),
        _trade_row(10.0, 1.0, "2026-01-05 09:22"),
        _trade_row(-5.0, -1.0, "2026-01-05 09:23"),
    ])
    metrics = _group_metrics(trades)
    assert metrics["profit_factor"] == 6.0
    assert metrics["win_rate_pct"] == 75.0
    assert metrics["total_r"] == 2.0


def test_group_metrics_max_drawdown_reflects_combined_equity_curve():
    trades = pd.DataFrame([
        _trade_row(10.0, 1.0, "2026-01-05 09:20"),
        _trade_row(-15.0, -1.5, "2026-01-05 09:21"),
        _trade_row(5.0, 0.5, "2026-01-05 09:22"),
    ])
    metrics = _group_metrics(trades)
    assert metrics["max_drawdown_points"] == 15.0


def test_group_metrics_empty_trades():
    metrics = _group_metrics(pd.DataFrame(columns=["symbol", "pnl_points", "r_multiple", "exit_time"]))
    assert metrics["trades"] == 0
    assert metrics["profit_factor"] is None


def test_summarize_groups_by_column():
    trades = pd.DataFrame([
        {**_trade_row(10.0, 1.0, "2026-01-05 09:20"), "side": "LONG"},
        {**_trade_row(-5.0, -1.0, "2026-01-05 09:21"), "side": "LONG"},
        {**_trade_row(-2.0, -0.4, "2026-01-05 09:22"), "side": "SHORT"},
    ])
    out = summarize(trades, "side").set_index("side")
    assert out.loc["LONG", "trades"] == 2
    assert out.loc["SHORT", "trades"] == 1
    assert out.loc["SHORT", "losses"] == 1


def test_build_diagnostic_trades_has_no_blocked_signals():
    # _signals_by_threshold already drops any row where the base signal carries a
    # blocker, so every trade this function produces must have empty blockers.
    trades = build_diagnostic_trades(_bars(), threshold=40.0, max_holding_bars=5)
    assert not trades.empty
    assert trades["blockers"].apply(len).eq(0).all()


def test_build_diagnostic_trades_matches_existing_threshold_sweep_aggregates():
    # Cross-check against the already-committed, causal run_threshold_sweep pipeline:
    # regenerating trade-level detail must not silently diverge from the aggregate
    # metrics that pipeline already reports for the same threshold and date split.
    bars = _bars()
    threshold = 60.0
    max_holding_bars = 5

    swept = run_threshold_sweep(bars, thresholds=(threshold,), max_holding_bars=max_holding_bars)
    trades = build_diagnostic_trades(bars, threshold=threshold, max_holding_bars=max_holding_bars)

    dates = sorted(pd.to_datetime(bars["timestamp"]).dt.date.unique())
    split_date = pd.Timestamp(dates[max(2, int(len(dates) * 0.60))]).date()

    train = trades[trades["signal_time"].apply(lambda t: t.date()) < split_date]
    oos = trades[trades["signal_time"].apply(lambda t: t.date()) >= split_date]

    train_metrics = _group_metrics(train)
    oos_metrics = _group_metrics(oos)

    row = swept.iloc[0]
    assert train_metrics["trades"] == row["train_trades"]
    assert train_metrics["win_rate_pct"] == row["train_win_rate_pct"]
    assert train_metrics["expectancy_r"] == row["train_expectancy_r"]
    assert oos_metrics["trades"] == row["oos_trades"]
    assert oos_metrics["win_rate_pct"] == row["oos_win_rate_pct"]
    assert oos_metrics["expectancy_r"] == row["oos_expectancy_r"]


def test_rvol_bucket_boundaries():
    assert rvol_bucket(0.5) == "<0.7 (blocker zone)"
    assert rvol_bucket(0.7) == "0.7-1.0"
    assert rvol_bucket(0.99) == "0.7-1.0"
    assert rvol_bucket(1.0) == "1.0-1.5"
    assert rvol_bucket(1.5) == "1.5-2.0"
    assert rvol_bucket(2.0) == "2.0+"
    assert rvol_bucket(float("nan")) == "unknown"


def test_adx_bucket_boundaries():
    assert adx_bucket(10.0) == "<15 (blocker zone)"
    assert adx_bucket(15.0) == "15-20"
    assert adx_bucket(22.0) == "20-25"
    assert adx_bucket(27.0) == "25-30"
    assert adx_bucket(35.0) == "30+"
    assert adx_bucket(float("nan")) == "unknown"


def test_vwap_distance_bucket_boundaries():
    assert vwap_distance_bucket(-3.0) == "<= -2.5% (blocker zone)"
    assert vwap_distance_bucket(-1.5) == "-2.5% to -1.0%"
    assert vwap_distance_bucket(-0.5) == "-1.0% to 0%"
    assert vwap_distance_bucket(0.5) == "0% to 1.0%"
    assert vwap_distance_bucket(2.0) == "1.0% to 2.5%"
    assert vwap_distance_bucket(3.0) == ">= 2.5% (blocker zone)"
    assert vwap_distance_bucket(float("nan")) == "unknown"


def test_build_diagnostic_trades_includes_rvol_adx_vwap_columns():
    trades = build_diagnostic_trades(_bars(), threshold=40.0, max_holding_bars=5)
    assert not trades.empty
    for col in ("relative_volume", "rvol_bucket", "adx14", "adx_bucket", "distance_from_vwap_pct", "vwap_distance_bucket"):
        assert col in trades.columns
    assert trades["rvol_bucket"].ne("unknown").all()


def test_cluster_significance_detects_a_real_difference():
    rng = np.random.default_rng(7)
    trades = pd.DataFrame({
        "bucket": ["A"] * 50 + ["B"] * 50,
        "r_multiple": np.concatenate([rng.normal(-2.0, 0.5, 50), rng.normal(0.5, 0.5, 50)]),
    })
    result = cluster_significance(trades, "bucket", min_n=30)
    assert set(result["bucket"]) == {"A", "B"}
    assert (result["p_value"] < 0.01).all()
    a_row = result[result["bucket"] == "A"].iloc[0]
    assert a_row["diff"] < 0


def test_cluster_significance_drops_groups_below_min_n():
    trades = pd.DataFrame({
        "bucket": ["A"] * 40 + ["B"] * 5,
        "r_multiple": [1.0] * 40 + [-1.0] * 5,
    })
    result = cluster_significance(trades, "bucket", min_n=30)
    assert list(result["bucket"]) == []
