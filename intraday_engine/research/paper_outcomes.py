from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from intraday_engine.storage.db import conn


OUTCOME_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_outcomes(
    observation_id VARCHAR PRIMARY KEY,
    evaluated_at TIMESTAMPTZ,
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    side VARCHAR,
    entry_price DOUBLE,
    exit_price DOUBLE,
    stop_loss DOUBLE,
    target DOUBLE,
    outcome VARCHAR,
    pnl_points DOUBLE,
    r_multiple DOUBLE,
    holding_bars INTEGER,
    mfe_points DOUBLE,
    mae_points DOUBLE
);
"""


def ensure_outcome_table() -> None:
    connection = conn()
    try:
        connection.execute(OUTCOME_SCHEMA)
    finally:
        connection.close()


def evaluate_trade(
    observation: dict,
    bars: pd.DataFrame,
    *,
    max_holding_bars: int = 30,
) -> dict | None:
    """Evaluate one completed paper signal using only bars after its signal bar.

    Returns None when insufficient future bars exist yet. Entry is the next
    bar open, while stop/target remain the levels recorded with the signal.
    If stop and target are both touched inside one bar, stop wins conservatively.
    """
    if max_holding_bars < 1:
        raise ValueError("max_holding_bars must be >= 1")
    if observation.get("signal_action") not in {"BUY", "SELL"}:
        return None
    if observation.get("stop_loss") is None or observation.get("target") is None:
        return None

    frame = bars.copy()
    if frame.empty:
        return None
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    signal_time = pd.Timestamp(observation["bar_time"])
    signal_positions = frame.index[frame["timestamp"] <= signal_time]
    if len(signal_positions) == 0:
        return None
    signal_index = int(signal_positions[-1])
    if signal_index + 1 >= len(frame):
        return None

    first_bar = frame.iloc[signal_index + 1]
    side = "LONG" if observation["signal_action"] == "BUY" else "SHORT"
    entry = float(first_bar["open"])
    stop = float(observation["stop_loss"])
    target = float(observation["target"])
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    max_end = min(signal_index + max_holding_bars, len(frame) - 1)
    mfe = 0.0
    mae = 0.0

    for position in range(signal_index + 1, max_end + 1):
        bar = frame.iloc[position]
        high = float(bar["high"])
        low = float(bar["low"])
        if side == "LONG":
            mfe = max(mfe, high - entry)
            mae = min(mae, low - entry)
            if float(bar["open"]) <= stop:
                exit_price, outcome = float(bar["open"]), "STOP_GAP"
            elif float(bar["open"]) >= target:
                exit_price, outcome = float(bar["open"]), "TARGET_GAP"
            elif low <= stop:
                exit_price, outcome = stop, "STOP"
            elif high >= target:
                exit_price, outcome = target, "TARGET"
            else:
                continue
        else:
            mfe = max(mfe, entry - low)
            mae = min(mae, entry - high)
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

        pnl = exit_price - entry if side == "LONG" else entry - exit_price
        return {
            "observation_id": observation["observation_id"],
            "evaluated_at": datetime.now().astimezone(),
            "entry_time": first_bar["timestamp"],
            "exit_time": bar["timestamp"],
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "stop_loss": stop,
            "target": target,
            "outcome": outcome,
            "pnl_points": pnl,
            "r_multiple": pnl / risk,
            "holding_bars": position - signal_index,
            "mfe_points": mfe,
            "mae_points": mae,
        }

    last = frame.iloc[max_end]
    exit_price = float(last["close"])
    pnl = exit_price - entry if side == "LONG" else entry - exit_price
    if max_end < signal_index + max_holding_bars:
        return None
    return {
        "observation_id": observation["observation_id"],
        "evaluated_at": datetime.now().astimezone(),
        "entry_time": first_bar["timestamp"],
        "exit_time": last["timestamp"],
        "side": side,
        "entry_price": entry,
        "exit_price": exit_price,
        "stop_loss": stop,
        "target": target,
        "outcome": "TIMEOUT",
        "pnl_points": pnl,
        "r_multiple": pnl / risk,
        "holding_bars": max_holding_bars,
        "mfe_points": mfe,
        "mae_points": mae,
    }


def _load_pending() -> list[dict]:
    connection = conn()
    try:
        rows = connection.execute(
            """
            SELECT observation_id, bar_time, symbol, signal_action,
                   entry_price, stop_loss, target
            FROM paper_observations
            WHERE status = 'SIGNAL_PENDING'
            ORDER BY observed_at
            """
        ).fetchall()
        columns = ["observation_id", "bar_time", "symbol", "signal_action", "entry_price", "stop_loss", "target"]
        return [dict(zip(columns, row)) for row in rows]
    finally:
        connection.close()


def _load_bars(symbol: str) -> pd.DataFrame:
    connection = conn()
    try:
        return connection.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = '1m'
            ORDER BY timestamp
            """,
            [symbol],
        ).df()
    finally:
        connection.close()


def persist_outcomes(outcomes: list[dict]) -> int:
    if not outcomes:
        return 0
    ensure_outcome_table()
    connection = conn()
    try:
        frame = pd.DataFrame(outcomes)
        connection.register("paper_outcomes_in", frame)
        connection.execute("INSERT OR IGNORE INTO paper_outcomes SELECT * FROM paper_outcomes_in")
        connection.execute(
            """
            UPDATE paper_observations
            SET status = 'EVALUATED'
            WHERE observation_id IN (SELECT observation_id FROM paper_outcomes_in)
            """
        )
        return len(outcomes)
    finally:
        connection.unregister("paper_outcomes_in")
        connection.close()


def evaluate_pending(*, max_holding_bars: int = 30) -> dict:
    pending = _load_pending()
    outcomes: list[dict] = []
    waiting = 0
    for observation in pending:
        result = evaluate_trade(observation, _load_bars(observation["symbol"]), max_holding_bars=max_holding_bars)
        if result is None:
            waiting += 1
        else:
            outcomes.append(result)
    evaluated = persist_outcomes(outcomes)
    return {"pending": len(pending), "evaluated": evaluated, "waiting": waiting}


def outcome_summary() -> dict:
    ensure_outcome_table()
    connection = conn()
    try:
        row = connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN pnl_points > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pnl_points <= 0 THEN 1 ELSE 0 END),
                   AVG(r_multiple),
                   SUM(CASE WHEN pnl_points > 0 THEN pnl_points ELSE 0 END),
                   -SUM(CASE WHEN pnl_points < 0 THEN pnl_points ELSE 0 END)
            FROM paper_outcomes
            """
        ).fetchone()
        total, wins, losses, avg_r, gross_profit, gross_loss = row
        return {
            "evaluated": int(total or 0),
            "wins": int(wins or 0),
            "losses": int(losses or 0),
            "win_rate": (float(wins) / float(total)) if total else 0.0,
            "avg_r": float(avg_r or 0.0),
            "profit_factor": (float(gross_profit) / float(gross_loss)) if gross_loss else None,
        }
    finally:
        connection.close()
