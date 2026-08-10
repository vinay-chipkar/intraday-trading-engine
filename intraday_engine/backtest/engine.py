from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import pandas as pd

from intraday_engine.signals.engine import TradeSignal


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    side: str
    signal_time: object
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    outcome: str
    pnl_points: float
    r_multiple: float
    holding_bars: int


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[BacktestTrade, ...]
    total_trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    profit_factor: float
    net_points: float
    max_drawdown_points: float
    expectancy_r: float


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")
    out = bars.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _fill_price(signal: TradeSignal, bar: pd.Series, slippage_points: float) -> float:
    raw = float(bar["open"])
    return raw + slippage_points if signal.side == "LONG" else raw - slippage_points


def _exit_price(side: str, price: float, slippage_points: float) -> float:
    return price - slippage_points if side == "LONG" else price + slippage_points


def _simulate_one(signal: TradeSignal, symbol_bars: pd.DataFrame, signal_index: int, *, max_holding_bars: int, slippage_points: float) -> BacktestTrade | None:
    if signal_index + 1 >= len(symbol_bars) or signal.stop_loss is None or signal.target is None:
        return None

    first_bar = symbol_bars.iloc[signal_index + 1]
    entry = _fill_price(signal, first_bar, slippage_points)
    stop = float(signal.stop_loss)
    target = float(signal.target)
    risk = abs(entry - stop)
    if not isfinite(entry) or not isfinite(stop) or not isfinite(target) or risk <= 0:
        return None

    side = signal.side
    for offset in range(max_holding_bars):
        position = signal_index + 1 + offset
        if position >= len(symbol_bars):
            break
        bar = symbol_bars.iloc[position]
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "LONG":
            if float(bar["open"]) <= stop:
                exit_price, outcome = float(bar["open"]), "STOP_GAP"
            elif float(bar["open"]) >= target:
                exit_price, outcome = float(bar["open"), "TARGET_GAP"
            elif low <= stop:
                exit_price, outcome = stop, "STOP"
            elif high >= target:
                exit_price, outcome = target, "TARGET"
            else:
                continue
        else:
            if float(bar["open"]) >= stop:
                exit_price, outcome = float(bar["open"]), "STOP_GAP"
            elif float(bar["open"]) <= target:
                exit_price, outcome = float(bar["open"]), "TARGET_GAP"
            elif high >= stop:
                exit_price, outcome = stop, "STOP"
            elif low <= target:
                exit_price, outcome = target, "TARGET"
            else:
                continue

        exit_price = _exit_price(side, exit_price, slippage_points)
        pnl = exit_price - entry if side == "LONG" else entry - exit_price
        return BacktestTrade(signal.symbol or "UNKNOWN", side, signal.event_time, first_bar["timestamp"], bar["timestamp"], entry, exit_price, stop, target, outcome, pnl, pnl / risk, offset + 1)

    last = symbol_bars.iloc[min(signal_index + max_holding_bars, len(symbol_bars) - 1)]
    exit_price = _exit_price(side, float(last["close"]), slippage_points)
    pnl = exit_price - entry if side == "LONG" else entry - exit_price
    return BacktestTrade(signal.symbol or "UNKNOWN", side, signal.event_time, first_bar["timestamp"], last["timestamp"], entry, exit_price, stop, target, "TIMEOUT", pnl, pnl / risk, max_holding_bars)


def _metrics(trades: list[BacktestTrade]) -> BacktestResult:
    if not trades:
        return BacktestResult((), 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = sum(t.pnl_points > 0 for t in trades)
    losses = sum(t.pnl_points <= 0 for t in trades)
    timeouts = sum(t.outcome == "TIMEOUT" for t in trades)
    gross_profit = sum(t.pnl_points for t in trades if t.pnl_points > 0)
    gross_loss = -sum(t.pnl_points for t in trades if t.pnl_points < 0)

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in trades:
        equity += trade.pnl_points
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return BacktestResult(tuple(trades), len(trades), wins, losses, timeouts, wins / len(trades), gross_profit / gross_loss if gross_loss else float("inf"), sum(t.pnl_points for t in trades), max_drawdown, sum(t.r_multiple for t in trades) / len(trades))


def backtest_signals(signals: Iterable[TradeSignal], bars: pd.DataFrame, *, max_holding_bars: int = 30, slippage_points: float = 0.0) -> BacktestResult:
    """Simulate completed-bar signals with next-bar execution."""
    if max_holding_bars < 1:
        raise ValueError("max_holding_bars must be >= 1")
    if slippage_points < 0:
        raise ValueError("slippage_points must be >= 0")

    frame = _validate_bars(bars)
    grouped = {symbol: group.reset_index(drop=True) for symbol, group in frame.groupby("symbol", sort=False)}
    ordered_signals = sorted((s for s in signals if s.action in {"BUY", "SELL"}), key=lambda signal: (str(signal.symbol), signal.event_time))
    active_until: dict[str, object] = {}
    trades: list[BacktestTrade] = []

    for signal in ordered_signals:
        if not signal.symbol or signal.event_time is None:
            continue
        symbol_bars = grouped.get(signal.symbol)
        if symbol_bars is None:
            continue
        if signal.symbol in active_until and signal.event_time <= active_until[signal.symbol]:
            continue

        timestamps = symbol_bars["timestamp"]
        signal_position = timestamps.searchsorted(pd.Timestamp(signal.event_time), side="right") - 1
        if signal_position < 0:
            continue

        trade = _simulate_one(signal, symbol_bars, int(signal_position), max_holding_bars=max_holding_bars, slippage_points=slippage_points)
        if trade is None:
            continue
        trades.append(trade)
        active_until[signal.symbol] = trade.exit_time

    trades.sort(key=lambda trade: (trade.exit_time, trade.symbol))
    return _metrics(trades)
