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

from intraday_engine.backtest.engine import backtest_signals
from intraday_engine.research.stats import (
    _group_metrics,
    adx_bucket,
    cluster_significance,
    entry_time_bucket,
    holding_bucket,
    rvol_bucket,
    score_bucket,
    summarize,
    vwap_distance_bucket,
)
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


def component_breakdown(trades: pd.DataFrame, component: str) -> pd.DataFrame:
    """Compare a component's support-vs-not-support subgroups within `trades`."""
    col = f"comp_{component}"
    out = summarize(trades, col)
    out = out.rename(columns={col: "supports_trade"})
    return out.sort_values("supports_trade", ascending=False).reset_index(drop=True)
