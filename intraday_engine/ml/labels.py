from __future__ import annotations

import pandas as pd


def _validate(df: pd.DataFrame) -> None:
    required = {"high", "low"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def label_forward_path(df: pd.DataFrame, entry: float, stop: float, target: float, *, side: str = "LONG", horizon: int = 30) -> dict:
    """Label only the path after entry; stop wins when both are touched in one OHLC bar."""
    _validate(df)
    if entry <= 0 or horizon < 1:
        raise ValueError("entry must be positive and horizon must be >= 1")
    side = side.upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")

    future = df.iloc[:horizon].copy()
    if future.empty:
        return {"target_hit_first": False, "mfe_pct": 0.0, "mae_pct": 0.0, "bars_to_exit": None, "exit_reason": "NO_FUTURE_DATA", "label": 0}

    target_hit = stop_hit = None
    for position, (_, row) in enumerate(future.iterrows(), start=1):
        hit_target = float(row["high"]) >= target if side == "LONG" else float(row["low"]) <= target
        hit_stop = float(row["low"]) <= stop if side == "LONG" else float(row["high"]) >= stop
        if hit_stop:
            stop_hit = position
            break
        if hit_target:
            target_hit = position
            break

    if stop_hit is not None:
        target_first, exit_reason, bars_to_exit = False, "STOP", stop_hit
    elif target_hit is not None:
        target_first, exit_reason, bars_to_exit = True, "TARGET", target_hit
    else:
        target_first, exit_reason, bars_to_exit = False, "TIMEOUT", len(future)

    if side == "LONG":
        mfe_pct = (float(future["high"].max()) - entry) / entry * 100.0
        mae_pct = (float(future["low"].min()) - entry) / entry * 100.0
    else:
        mfe_pct = (entry - float(future["low"].min())) / entry * 100.0
        mae_pct = (entry - float(future["high"].max())) / entry * 100.0

    return {"target_hit_first": target_first, "mfe_pct": float(mfe_pct), "mae_pct": float(mae_pct), "bars_to_exit": int(bars_to_exit), "exit_reason": exit_reason, "label": int(target_first)}


def build_forward_labels(df: pd.DataFrame, entries: pd.DataFrame, *, horizon: int = 30) -> pd.DataFrame:
    """Build labels from candles strictly after each timestamped entry."""
    _validate(df)
    required = {"timestamp", "entry", "stop", "target", "side"}
    missing = required.difference(entries.columns)
    if missing:
        raise ValueError(f"Missing entry columns: {sorted(missing)}")

    prices = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(prices["timestamp"], utc=True)
    results = []
    for _, entry_row in entries.iterrows():
        timestamp = pd.Timestamp(entry_row["timestamp"])
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        future = prices.loc[timestamps > timestamp]
        result = entry_row.to_dict()
        result.update(label_forward_path(future, float(entry_row["entry"]), float(entry_row["stop"]), float(entry_row["target"]), side=str(entry_row["side"]), horizon=horizon))
        results.append(result)
    return pd.DataFrame(results)
