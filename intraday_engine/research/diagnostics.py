"""Trade-level diagnostic decomposition of the existing point-in-time backtest.

This module does not run a new simulation and does not change any signal rule
or strategy parameter. It reuses the exact causal pipeline already exercised by
`threshold_sensitivity.run_threshold_sweep` (`_prepare_symbol` for point-in-time
enrichment, `_signals_by_threshold` for signal generation, `backtest_signals` for
next-bar execution) and joins the resulting trades back to the row/signal that
produced them, so per-trade score, component, and timing detail is available for
analysis without re-deriving any trading logic.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from intraday_engine.backtest.engine import backtest_signals
from intraday_engine.research.threshold_sensitivity import _prepare_symbol, _signals_by_threshold

COMPONENT_NAMES = (
    "ema_trend", "market_structure", "ema_alignment", "vwap", "rsi",
    "macd", "adx", "rvol", "orb", "candlestick",
)


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if value != value else value  # NaN check


def _component_support(row: pd.Series, bullish: bool) -> dict[str, bool]:
    """Reproduce signals/engine.py::_score's per-component bullish/bearish conditions.

    A component "supports" a trade when its condition points the same direction as
    the trade that was actually taken (BUY -> bullish, SELL -> bearish). Anything
    else (opposite direction, or the component being neutral for that bar) counts
    as not supporting -- this mirrors exactly what `_score` checked at signal time.
    """
    close = _num(row, "close")
    ema9, ema20, ema50 = _num(row, "ema9"), _num(row, "ema20"), _num(row, "ema50")
    vwap = _num(row, "vwap")
    rsi = _num(row, "rsi14", 50.0)
    adx = _num(row, "adx14")
    plus_di, minus_di = _num(row, "plus_di14"), _num(row, "minus_di14")
    macd_hist = _num(row, "macd_histogram")
    rvol = _num(row, "relative_volume")
    trend = str(row.get("trend", "SIDEWAYS"))
    structure = str(row.get("structure_trend", trend))

    if bullish:
        return {
            "ema_trend": trend == "UPTREND",
            "market_structure": structure == "UPTREND",
            "ema_alignment": close > ema9 > ema20 > ema50 > 0,
            "vwap": vwap > 0 and close > vwap,
            "rsi": 52 <= rsi <= 70,
            "macd": macd_hist > 0,
            "adx": adx >= 25 and plus_di >= minus_di,
            "rvol": rvol >= 1.5 and close >= vwap,
            "orb": bool(row.get("opening_range_breakout")),
            "candlestick": bool(row.get("hammer") or row.get("bullish_engulfing") or row.get("morning_star")),
        }
    return {
        "ema_trend": trend == "DOWNTREND",
        "market_structure": structure == "DOWNTREND",
        "ema_alignment": 0 < close < ema9 < ema20 < ema50,
        "vwap": vwap > 0 and close < vwap,
        "rsi": 30 <= rsi <= 48,
        "macd": macd_hist < 0,
        "adx": adx >= 25 and plus_di < minus_di,
        "rvol": rvol >= 1.5 and close < vwap,
        "orb": bool(row.get("opening_range_breakdown")),
        "candlestick": bool(row.get("shooting_star") or row.get("bearish_engulfing") or row.get("evening_star")),
    }


def score_bucket(abs_score: float) -> str:
    if abs_score >= 70:
        return "70+"
    if abs_score >= 60:
        return "60-69"
    if abs_score >= 50:
        return "50-59"
    return "40-49"


_ENTRY_TIME_BUCKETS = (
    (555, 600, "09:15-10:00"),
    (600, 660, "10:00-11:00"),
    (660, 720, "11:00-12:00"),
    (720, 780, "12:00-13:00"),
    (780, 840, "13:00-14:00"),
    (840, 930, "14:00-15:30"),
)


def entry_time_bucket(entry_time_ist: pd.Timestamp) -> str:
    minutes = entry_time_ist.hour * 60 + entry_time_ist.minute
    for lo, hi, label in _ENTRY_TIME_BUCKETS:
        if lo <= minutes < hi:
            return label
    return "OUTSIDE_SESSION"


def holding_bucket(bars: int, max_holding_bars: int = 30, width: int = 5) -> str:
    edges = list(range(0, max_holding_bars + 1, width))
    if edges[-1] < max_holding_bars:
        edges.append(max_holding_bars)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo < bars <= hi:
            return f"{lo + 1}-{hi}"
    return f"{max_holding_bars + 1}+"


def rvol_bucket(rvol: float) -> str:
    """Buckets aligned with _score's own RVOL thresholds (0.7 blocker floor, 1.5 boost)."""
    if rvol != rvol:  # NaN
        return "unknown"
    if rvol < 0.7:
        return "<0.7 (blocker zone)"
    if rvol < 1.0:
        return "0.7-1.0"
    if rvol < 1.5:
        return "1.0-1.5"
    if rvol < 2.0:
        return "1.5-2.0"
    return "2.0+"


def adx_bucket(adx: float) -> str:
    """Buckets aligned with _score's own ADX thresholds (15 blocker floor, 25 boost)."""
    if adx != adx:  # NaN
        return "unknown"
    if adx < 15:
        return "<15 (blocker zone)"
    if adx < 20:
        return "15-20"
    if adx < 25:
        return "20-25"
    if adx < 30:
        return "25-30"
    return "30+"


def vwap_distance_bucket(distance_pct: float) -> str:
    """Buckets aligned with _score's own VWAP-distance blocker threshold (+/-2.5%)."""
    if distance_pct != distance_pct:  # NaN
        return "unknown"
    if distance_pct <= -2.5:
        return "<= -2.5% (blocker zone)"
    if distance_pct < -1.0:
        return "-2.5% to -1.0%"
    if distance_pct < 0:
        return "-1.0% to 0%"
    if distance_pct < 1.0:
        return "0% to 1.0%"
    if distance_pct < 2.5:
        return "1.0% to 2.5%"
    return ">= 2.5% (blocker zone)"


def build_diagnostic_trades(
    candles: pd.DataFrame,
    *,
    threshold: float = 40.0,
    max_holding_bars: int = 30,
    slippage_points: float = 0.0,
) -> pd.DataFrame:
    """Regenerate one threshold's trades via the existing causal pipeline, enriched
    with the score/component/timing detail needed for diagnostics.

    `threshold` should be the lowest threshold of interest: `_signals_by_threshold`
    qualifies a signal whenever `abs(score) >= threshold`, so using the lowest
    threshold (40) yields the superset of every higher threshold's trades and lets
    score-bucket analysis be done from one consistent, non-overlapping population.
    """
    records: list[dict] = []
    for symbol, group in candles.groupby("symbol", sort=True):
        prepared = _prepare_symbol(group)
        signals = _signals_by_threshold(prepared, (threshold,))[threshold]
        if not signals:
            continue
        signal_by_key = {(s.symbol, pd.Timestamp(s.event_time)): s for s in signals}
        rows_by_ts = prepared.set_index("timestamp", drop=False)

        result = backtest_signals(
            signals, prepared, max_holding_bars=max_holding_bars, slippage_points=slippage_points
        )
        for trade in result.trades:
            ts = pd.Timestamp(trade.signal_time)
            signal = signal_by_key[(trade.symbol, ts)]
            row = rows_by_ts.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            bullish = signal.action == "BUY"
            support = _component_support(row, bullish)
            entry_time = pd.Timestamp(trade.entry_time)
            entry_ist = entry_time.tz_convert("Asia/Kolkata") if entry_time.tzinfo is not None else entry_time
            rvol = _num(row, "relative_volume")
            adx = _num(row, "adx14")
            vwap_distance = _num(row, "distance_from_vwap_pct")
            records.append({
                "symbol": trade.symbol,
                "action": signal.action,
                "side": trade.side,
                "score": float(signal.score),
                "abs_score": abs(float(signal.score)),
                "score_bucket": score_bucket(abs(float(signal.score))),
                "signal_time": ts,
                "entry_time": entry_time,
                "exit_time": pd.Timestamp(trade.exit_time),
                "entry_time_bucket": entry_time_bucket(entry_ist),
                "holding_bars": trade.holding_bars,
                "holding_bucket": holding_bucket(trade.holding_bars, max_holding_bars),
                "outcome": trade.outcome,
                "pnl_points": trade.pnl_points,
                "r_multiple": trade.r_multiple,
                "blockers": signal.blockers,
                "relative_volume": rvol,
                "rvol_bucket": rvol_bucket(rvol),
                "adx14": adx,
                "adx_bucket": adx_bucket(adx),
                "distance_from_vwap_pct": vwap_distance,
                "vwap_distance_bucket": vwap_distance_bucket(vwap_distance),
                **{f"comp_{name}": support[name] for name in COMPONENT_NAMES},
            })
    return pd.DataFrame.from_records(records)


def _group_metrics(trades: pd.DataFrame) -> dict[str, float | int | None]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                "profit_factor": None, "expectancy_r": 0.0, "total_r": 0.0,
                "net_points": 0.0, "max_drawdown_points": 0.0}
    wins = int((trades["pnl_points"] > 0).sum())
    losses = int((trades["pnl_points"] <= 0).sum())
    gross_profit = float(trades.loc[trades["pnl_points"] > 0, "pnl_points"].sum())
    gross_loss = float(-trades.loc[trades["pnl_points"] < 0, "pnl_points"].sum())
    ordered = trades.sort_values(["exit_time", "symbol"])
    equity = ordered["pnl_points"].cumsum()
    peak = equity.cummax()
    drawdown = float((peak - equity).max())
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / n * 100.0, 3),
        "profit_factor": None if gross_loss == 0 else round(gross_profit / gross_loss, 4),
        "expectancy_r": round(float(trades["r_multiple"].mean()), 5),
        "total_r": round(float(trades["r_multiple"].sum()), 3),
        "net_points": round(float(trades["pnl_points"].sum()), 4),
        "max_drawdown_points": round(drawdown, 4),
    }


def summarize(trades: pd.DataFrame, group_cols: str | list[str] | None = None) -> pd.DataFrame:
    """Group-wise trade metrics (trades/wins/losses/win_rate/PF/expectancy/total_r/drawdown)."""
    if group_cols is None:
        return pd.DataFrame([_group_metrics(trades)])
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    rows = []
    for key, group in trades.groupby(group_cols, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, keys))
        row.update(_group_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def component_breakdown(trades: pd.DataFrame, component: str) -> pd.DataFrame:
    """Compare a component's support-vs-not-support subgroups within `trades`."""
    col = f"comp_{component}"
    out = summarize(trades, col)
    out = out.rename(columns={col: "supports_trade"})
    return out.sort_values("supports_trade", ascending=False).reset_index(drop=True)


def cluster_significance(trades: pd.DataFrame, group_col: str, min_n: int = 30) -> pd.DataFrame:
    """Welch's t-test of each group_col level's r_multiple against every other level's.

    This is a magnitude+significance screen for "is this cluster's R distribution
    different from the rest of the population", not a claim about a trading edge --
    it says nothing about win/loss economics beyond the R multiple itself, and with
    many levels tested at once the usual multiple-comparisons caveat applies (some
    p<0.05 results are expected by chance). Levels with fewer than `min_n` trades on
    either side of the split are dropped rather than reported with an unreliable
    p-value.
    """
    rows = []
    for level, group in trades.groupby(group_col, dropna=False):
        rest = trades[trades[group_col] != level]
        if len(group) < min_n or len(rest) < min_n:
            continue
        stat, pvalue = stats.ttest_ind(group["r_multiple"], rest["r_multiple"], equal_var=False)
        rows.append({
            group_col: level,
            "trades": len(group),
            "expectancy_r": round(float(group["r_multiple"].mean()), 5),
            "rest_expectancy_r": round(float(rest["r_multiple"].mean()), 5),
            "diff": round(float(group["r_multiple"].mean() - rest["r_multiple"].mean()), 5),
            "t_stat": round(float(stat), 3),
            "p_value": round(float(pvalue), 5),
        })
    columns = [group_col, "trades", "expectancy_r", "rest_expectancy_r", "diff", "t_stat", "p_value"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
