from __future__ import annotations

import pandas as pd

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]


def normalize_candles(frame: pd.DataFrame, *, instrument_key: str, symbol: str, interval: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["instrument_key", "symbol", "timestamp", "interval", "open", "high", "low", "close", "volume", "open_interest"])
    missing = set(CANDLE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    output = frame[CANDLE_COLUMNS].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
    for column in ["open", "high", "low", "close", "volume", "open_interest"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["timestamp", "open", "high", "low", "close"])
    output = output[(output["high"] >= output[["open", "close", "low"]].max(axis=1)) & (output["low"] <= output[["open", "close", "high"]].min(axis=1)) & (output["volume"].fillna(0) >= 0)]
    output = output.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    output.insert(0, "instrument_key", instrument_key)
    output.insert(1, "symbol", symbol.upper())
    output.insert(3, "interval", interval)
    return output[["instrument_key", "symbol", "timestamp", "interval", "open", "high", "low", "close", "volume", "open_interest"]]


def quality_report(frame: pd.DataFrame) -> dict[str, int | bool]:
    if frame is None or frame.empty:
        return {"rows": 0, "duplicates": 0, "invalid_ohlc": 0, "negative_volume": 0, "null_timestamps": 0, "monotonic": True}
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_ohlc = ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).sum()
    return {
        "rows": int(len(frame)),
        "duplicates": int(frame["timestamp"].duplicated().sum()),
        "invalid_ohlc": int(invalid_ohlc),
        "negative_volume": int((pd.to_numeric(frame["volume"], errors="coerce") < 0).sum()),
        "null_timestamps": int(timestamps.isna().sum()),
        "monotonic": bool(timestamps.dropna().is_monotonic_increasing),
    }
