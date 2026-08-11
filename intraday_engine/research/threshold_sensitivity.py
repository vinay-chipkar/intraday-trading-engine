from __future__ import annotations

from typing import Iterable

import pandas as pd

from intraday_engine.backtest.engine import BacktestResult, backtest_signals
from intraday_engine.patterns.candles import add_candle_patterns
from intraday_engine.signals.engine import SignalConfig, generate_signal
from intraday_engine.technical.indicators import add_indicators


def _prepare_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("timestamp").reset_index(drop=True).copy()
    out = add_indicators(out)
    out = add_candle_patterns(out)
    # Structural levels use completed bars only; no centered/future windows.
    out["support"] = out["low"].shift(1).rolling(20, min_periods=20).min()
    out["resistance"] = out["high"].shift(1).rolling(20, min_periods=20).max()
    out["structure_trend"] = out["trend"]
    out["double_bottom"] = False
    out["double_top"] = False
    return out


def _signals_for_threshold(frame: pd.DataFrame, threshold: float) -> list:
    signals = []
    config = SignalConfig(buy_threshold=threshold, sell_threshold=-threshold)
    required = ("ema50", "atr14", "vwap", "support", "resistance")
    for row in frame.to_dict("records"):
        if any(pd.isna(row.get(name)) for name in required):
            continue
        signal = generate_signal(
            row,
            market_score=0.0,
            config=config,
            symbol=str(row["symbol"]),
            event_time=row["timestamp"],
        )
        if signal.action in {"BUY", "SELL"}:
            signals.append(signal)
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


def _partition(result: BacktestResult, split_date: pd.Timestamp, train: bool) -> dict:
    if train:
        trades = [t for t in result.trades if pd.Timestamp(t.signal_time) < split_date]
    else:
        trades = [t for t in result.trades if pd.Timestamp(t.signal_time) >= split_date]
    return _metrics(trades)


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

    prepared = [_prepare_symbol(group) for _, group in candles.groupby("symbol", sort=True)]
    frame = pd.concat(prepared, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    dates = sorted(pd.to_datetime(frame["timestamp"]).dt.date.unique())
    if len(dates) < 4:
        raise ValueError("Need at least 4 trading dates for train/OOS research")
    split_date = pd.Timestamp(dates[max(2, int(len(dates) * 0.60))])

    rows = []
    for threshold in thresholds:
        signals = []
        for _, group in frame.groupby("symbol", sort=True):
            signals.extend(_signals_for_threshold(group, float(threshold)))
        result = backtest_signals(signals, frame, max_holding_bars=max_holding_bars, slippage_points=slippage_points)
        train = _partition(result, split_date, True)
        oos = _partition(result, split_date, False)
        row = {"threshold": float(threshold), "split_date": split_date.date().isoformat()}
        row.update({f"train_{k}": v for k, v in train.items()})
        row.update({f"oos_{k}": v for k, v in oos.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
