from __future__ import annotations

import pandas as pd

from intraday_engine.market.session import MARKET_CLOSE, MARKET_OPEN


CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
]


def normalize_candles(
    frame: pd.DataFrame,
    *,
    instrument_key: str,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    """Validate and normalize an Upstox candle frame for persistence."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "instrument_key", "symbol", "timestamp", "interval",
                "open", "high", "low", "close", "volume", "open_interest",
            ]
        )

    missing = set(CANDLE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    output = frame[CANDLE_COLUMNS].copy()
    output["timestamp"] = pd.to_datetime(
        output["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert("Asia/Kolkata")

    numeric = ["open", "high", "low", "close", "volume", "open_interest"]
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=["timestamp", "open", "high", "low", "close"])
    output = output[
        (output["high"] >= output[["open", "close", "low"]].max(axis=1))
        & (output["low"] <= output[["open", "close", "high"]].min(axis=1))
        & (output["volume"].fillna(0) >= 0)
    ]

    output = (
        output.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    output.insert(0, "instrument_key", instrument_key)
    output.insert(1, "symbol", symbol.upper())
    output.insert(3, "interval", interval)
    return output[
        [
            "instrument_key", "symbol", "timestamp", "interval",
            "open", "high", "low", "close", "volume", "open_interest",
        ]
    ]


def _session_gap_count(timestamps: pd.Series) -> int:
    """Count missing 1-minute boundaries strictly between the earliest and
    latest timestamp present -- an internal consistency check on what was
    actually received (independent of "now"/staleness, which is a separate
    concern). Returns 0 if fewer than two distinct minutes are present."""
    clean = timestamps.dropna().dt.floor("min").drop_duplicates().sort_values()
    if len(clean) < 2:
        return 0
    expected = pd.date_range(clean.iloc[0], clean.iloc[-1], freq="1min")
    return int(len(expected) - len(clean))


def _outside_session_count(timestamps: pd.Series) -> int:
    """Count candles whose local (Asia/Kolkata) time falls outside the NSE
    cash-segment session (09:15-15:30), a session-boundary violation."""
    local = timestamps.dropna()
    if local.empty:
        return 0
    if local.dt.tz is None:
        local = local.dt.tz_localize("UTC")
    local = local.dt.tz_convert("Asia/Kolkata")
    times = local.dt.time
    return int(((times < MARKET_OPEN) | (times > MARKET_CLOSE)).sum())


def quality_report(frame: pd.DataFrame) -> dict[str, int | bool]:
    """Return deterministic quality metrics without mutating the input."""
    if frame is None or frame.empty:
        return {
            "rows": 0,
            "duplicates": 0,
            "invalid_ohlc": 0,
            "negative_volume": 0,
            "null_timestamps": 0,
            "monotonic": True,
            "session_gaps": 0,
            "outside_session": 0,
        }

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    ).sum()
    return {
        "rows": int(len(frame)),
        "duplicates": int(frame["timestamp"].duplicated().sum()),
        "invalid_ohlc": int(invalid_ohlc),
        "negative_volume": int((pd.to_numeric(frame["volume"], errors="coerce") < 0).sum()),
        "null_timestamps": int(timestamps.isna().sum()),
        "monotonic": bool(timestamps.dropna().is_monotonic_increasing),
        "session_gaps": _session_gap_count(timestamps),
        "outside_session": _outside_session_count(timestamps),
    }
