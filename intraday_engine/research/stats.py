"""Backend-agnostic trade/observation statistics shared by every diagnostics
surface in this codebase (backtest trades in research/diagnostics.py, paper
observations in research/paper_diagnostics.py).

Deliberately has no dependency on the backtest engine, DuckDB, or any
specific data source: every function here operates on a plain DataFrame with
a `pnl_points`/`r_multiple` column (plus `exit_time`/`symbol` for drawdown
ordering), so the exact same significance-testing methodology is used
everywhere instead of two implementations that could silently drift apart.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def score_bucket(abs_score: float) -> str:
    if abs_score >= 70:
        return "70+"
    if abs_score >= 60:
        return "60-69"
    if abs_score >= 50:
        return "50-59"
    return "40-49"


def candidate_score_bucket(score: float) -> str:
    """Buckets for the scanner's 0-100 candidate_score (not gated at 40 like signal_score)."""
    if score != score:  # NaN
        return "unknown"
    if score < 30:
        return "<30"
    if score < 50:
        return "30-49"
    if score < 70:
        return "50-69"
    return "70+"


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


def rsi_bucket(rsi: float) -> str:
    if rsi != rsi:  # NaN
        return "unknown"
    if rsi < 30:
        return "<30 (oversold)"
    if rsi < 48:
        return "30-48 (bearish)"
    if rsi <= 52:
        return "48-52 (neutral)"
    if rsi <= 70:
        return "52-70 (bullish)"
    return ">70 (overbought)"


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
