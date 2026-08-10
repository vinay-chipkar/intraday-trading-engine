from __future__ import annotations

import pandas as pd

from intraday_engine.backtest.engine import BacktestResult
from intraday_engine.signals.engine import TradeSignal


def trade_diagnostics(
    result: BacktestResult,
    signals: list[TradeSignal],
) -> pd.DataFrame:
    """Join completed backtest trades with the signal metadata that created them."""
    signal_map = {
        (str(signal.symbol), pd.Timestamp(signal.event_time)): signal
        for signal in signals
        if signal.symbol and signal.event_time is not None
    }

    rows: list[dict[str, object]] = []
    for trade in result.trades:
        signal = signal_map.get((trade.symbol, pd.Timestamp(trade.signal_time)))
        rows.append(
            {
                "symbol": trade.symbol,
                "side": trade.side,
                "signal_time": trade.signal_time,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "outcome": trade.outcome,
                "entry_price": trade.entry_price,
                "stop_loss": trade.stop_loss,
                "target": trade.target,
                "exit_price": trade.exit_price,
                "pnl_points": trade.pnl_points,
                "r_multiple": trade.r_multiple,
                "holding_bars": trade.holding_bars,
                "signal_score": signal.score if signal else None,
                "confidence": signal.confidence if signal else None,
                "reasons": " | ".join(signal.reasons) if signal else "",
            }
        )

    return pd.DataFrame(rows)


def summarize_diagnostics(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return side, score-band, and reason summaries for completed trades."""
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    by_side = (
        trades.groupby(["symbol", "side"], dropna=False)
        .agg(
            trades=("pnl_points", "size"),
            wins=("pnl_points", lambda s: int((s > 0).sum())),
            net_points=("pnl_points", "sum"),
            expectancy_r=("r_multiple", "mean"),
            avg_score=("signal_score", "mean"),
        )
        .reset_index()
    )
    by_side["win_rate"] = by_side["wins"] / by_side["trades"]

    bands = [-float("inf"), 60, 65, 70, 75, 80, 85, 90, float("inf")]
    labels = ["<60", "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90+"]
    scored = trades.dropna(subset=["signal_score"]).copy()
    scored["score_band"] = pd.cut(scored["signal_score"], bins=bands, labels=labels, right=False)
    by_score = (
        scored.groupby("score_band", observed=False)
        .agg(
            trades=("pnl_points", "size"),
            wins=("pnl_points", lambda s: int((s > 0).sum())),
            net_points=("pnl_points", "sum"),
            expectancy_r=("r_multiple", "mean"),
        )
        .reset_index()
    )
    by_score["win_rate"] = by_score["wins"] / by_score["trades"].where(by_score["trades"] > 0)

    reason_rows: list[dict[str, object]] = []
    for _, trade in trades.iterrows():
        for reason in filter(None, str(trade["reasons"]).split(" | ")):
            reason_rows.append(
                {
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "reason": reason,
                    "pnl_points": trade["pnl_points"],
                    "r_multiple": trade["r_multiple"],
                }
            )
    reasons = pd.DataFrame(reason_rows)
    if reasons.empty:
        by_reason = pd.DataFrame()
    else:
        by_reason = (
            reasons.groupby(["symbol", "side", "reason"])
            .agg(
                occurrences=("pnl_points", "size"),
                net_points=("pnl_points", "sum"),
                expectancy_r=("r_multiple", "mean"),
                win_rate=("pnl_points", lambda s: float((s > 0).mean())),
            )
            .reset_index()
            .sort_values(["symbol", "side", "net_points"], ascending=[True, True, False])
        )

    return by_side, by_score, by_reason
