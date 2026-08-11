from __future__ import annotations

from typing import Iterable

import pandas as pd

from intraday_engine.backtest.engine import backtest_signals
from intraday_engine.patterns.candles import add_candle_patterns
from intraday_engine.signals.engine import SignalConfig, TradeSignal, generate_signal
from intraday_engine.strategy.point_in_time import _confirmed_pivots, pivot_structure
from intraday_engine.technical.indicators import add_indicators


def _prepare_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("timestamp").reset_index(drop=True).copy()
    out = add_indicators(out)
    out = add_candle_patterns(out)
    # Structural levels use completed bars only; no centered/future windows.
    out["support"] = out["low"].shift(1).rolling(20, min_periods=20).min()
    out["resistance"] = out["high"].shift(1).rolling(20, min_periods=20).max()
    # "market structure" is the live pipeline's causal pivot-confirmed trend
    # (strategy/point_in_time.py::pivot_structure), not a copy of the EMA-order
    # trend above -- the two are independent signals in the live path and must
    # stay independent here too.
    confirmed_high, confirmed_low = _confirmed_pivots(out, 3, 3)
    out["structure_trend"] = pivot_structure(confirmed_high, confirmed_low, out["close"])["trend"]
    out["double_bottom"] = False
    out["double_top"] = False
    return out


def _base_signal(row: dict) -> TradeSignal | None:
    """Evaluate the expensive point-in-time signal logic once per bar.

    Thresholds only decide whether an already-computed score qualifies.  The
    score, blockers and structural stop do not depend on the threshold, so
    recomputing generate_signal for every threshold was unnecessary work.
    """
    required = ("ema50", "atr14", "vwap", "support", "resistance")
    if any(pd.isna(row.get(name)) for name in required):
        return None
    return generate_signal(
        row,
        market_score=0.0,
        config=SignalConfig(buy_threshold=0.0, sell_threshold=0.0),
        symbol=str(row["symbol"]),
        event_time=row["timestamp"],
    )


def _signals_by_threshold(frame: pd.DataFrame, thresholds: tuple[float, ...]) -> dict[float, list[TradeSignal]]:
    """Generate signals once per bar and fan them out to qualifying thresholds."""
    signals = {threshold: [] for threshold in thresholds}
    for row in frame.to_dict("records"):
        base = _base_signal(row)
        if base is None or base.action == "NO_TRADE" or base.blockers:
            continue
        score = float(base.score)
        for threshold in thresholds:
            if abs(score) >= threshold:
                signals[threshold].append(base)
    return signals


def _metrics(trades: list) -> dict[str, float | int | None]:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0, "win_rate_pct": 0.0,
                "profit_factor": None, "net_points": 0.0, "max_drawdown_points": 0.0, "expectancy_r": 0.0}
    wins = sum(t.pnl_points > 0 for t in trades)
    losses = sum(t.pnl_points <= 0 for t in trades)
    timeouts = sum(t.outcome == "TIMEOUT" for t in trades)
    gross_profit = sum(t.pnl_points for t in trades if t.pnl_points > 0)
    gross_loss = -sum(t.pnl_points for t in trades if t.pnl_points < 0)
    equity = peak = drawdown = 0.0
    for trade in sorted(trades, key=lambda t: (t.exit_time, t.symbol)):
        equity += trade.pnl_points
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trades": len(trades), "wins": wins, "losses": losses, "timeouts": timeouts,
        "win_rate_pct": round(wins / len(trades) * 100.0, 3),
        "profit_factor": None if gross_loss == 0 else round(gross_profit / gross_loss, 4),
        "net_points": round(sum(t.pnl_points for t in trades), 4),
        "max_drawdown_points": round(drawdown, 4),
        "expectancy_r": round(sum(t.r_multiple for t in trades) / len(trades), 5),
    }


def _partition(trades: list, split_date: pd.Timestamp, train: bool) -> dict:
    # Compare calendar dates, not full timestamps: split_date is derived from
    # .dt.date (always tz-naive), while a trade's signal_time carries whatever
    # tz-awareness the source candles had (real Upstox data is tz-aware) -
    # comparing the raw Timestamps raises "Cannot compare tz-naive and
    # tz-aware timestamps" whenever the two disagree.
    boundary = split_date.date()
    if train:
        selected = [t for t in trades if pd.Timestamp(t.signal_time).date() < boundary]
    else:
        selected = [t for t in trades if pd.Timestamp(t.signal_time).date() >= boundary]
    return _metrics(selected)


def run_threshold_sweep(
    candles: pd.DataFrame,
    thresholds: Iterable[float] = (40, 45, 50, 55, 60, 65, 70),
    *,
    max_holding_bars: int = 30,
    slippage_points: float = 0.0,
) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values:
        raise ValueError("At least one threshold is required")

    prepared = [_prepare_symbol(group) for _, group in candles.groupby("symbol", sort=True)]
    frame = pd.concat(prepared, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    dates = sorted(pd.to_datetime(frame["timestamp"]).dt.date.unique())
    if len(dates) < 4:
        raise ValueError("Need at least 4 trading dates for train/OOS research")
    split_date = pd.Timestamp(dates[max(2, int(len(dates) * 0.60))])

    # Backtest per symbol (cheaper: avoids scanning the full multi-symbol
    # frame per signal) but pool the resulting trades per threshold before
    # computing metrics, so profit_factor/drawdown reflect the true combined
    # trade stream rather than an approximation derived from per-symbol nets.
    trades_by_threshold: dict[float, list] = {threshold: [] for threshold in threshold_values}
    for symbol, group in frame.groupby("symbol", sort=True):
        print(f"Processing {symbol} ({len(group):,} bars)", flush=True)
        by_threshold = _signals_by_threshold(group, threshold_values)
        for threshold in threshold_values:
            result = backtest_signals(
                by_threshold[threshold],
                group,
                max_holding_bars=max_holding_bars,
                slippage_points=slippage_points,
            )
            trades_by_threshold[threshold].extend(result.trades)

    rows = []
    for threshold in threshold_values:
        trades = trades_by_threshold[threshold]
        train = _partition(trades, split_date, True)
        oos = _partition(trades, split_date, False)
        rows.append({
            "threshold": threshold,
            "split_date": split_date.date().isoformat(),
            **{f"train_{k}": v for k, v in train.items()},
            **{f"oos_{k}": v for k, v in oos.items()},
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
