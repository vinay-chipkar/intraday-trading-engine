from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    return result.fillna(50.0).clip(0, 100)


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    session = timestamps.dt.tz_convert("Asia/Kolkata").dt.date if getattr(timestamps.dt, "tz", None) else timestamps.dt.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]
    return pv.groupby(session).cumsum() / df["volume"].groupby(session).cumsum().replace(0, np.nan)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, stddev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(period, min_periods=period).mean()
    deviation = series.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + stddev * deviation
    lower = middle - stddev * deviation
    return middle, upper, lower


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]
    previous_high = high.shift(1)
    previous_low = low.shift(1)
    up_move = high - previous_high
    down_move = previous_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = true_range(df)
    atr_value = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_value.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_value.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_value = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_value, plus_di, minus_di


def _session_opening_range(df: pd.DataFrame, minutes: int = 15) -> tuple[pd.Series, pd.Series]:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        local = timestamps.dt.tz_convert("Asia/Kolkata")
    else:
        local = timestamps
    session_date = local.dt.date
    session_minutes = (local.dt.hour * 60 + local.dt.minute) - (9 * 60 + 15)
    opening = session_minutes.between(0, minutes - 1)
    high = df["high"].where(opening).groupby(session_date).cummax()
    low = df["low"].where(opening).groupby(session_date).cummin()
    # Carry the completed opening range forward through the rest of the session.
    return high.groupby(session_date).ffill(), low.groupby(session_date).ffill()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic technical features without using future rows."""
    _validate_ohlcv(df)
    out = df.copy().sort_values("timestamp").reset_index(drop=True)

    out["ema9"] = ema(out["close"], 9)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi14"] = rsi(out["close"], 14)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / out["close"].replace(0, np.nan) * 100.0
    out["vwap"] = vwap(out)

    macd_line, signal_line, histogram = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_histogram"] = histogram

    bb_middle, bb_upper, bb_lower = bollinger_bands(out["close"])
    out["bb_middle"] = bb_middle
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_width_pct"] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan) * 100.0

    adx_value, plus_di, minus_di = adx(out)
    out["adx14"] = adx_value
    out["plus_di14"] = plus_di
    out["minus_di14"] = minus_di

    timestamps = pd.to_datetime(out["timestamp"], errors="coerce")
    local = timestamps.dt.tz_convert("Asia/Kolkata") if getattr(timestamps.dt, "tz", None) is not None else timestamps
    session = local.dt.date
    out["session_volume"] = out["volume"].groupby(session).cumsum()
    out["volume_sma20"] = out["volume"].rolling(20, min_periods=5).mean()
    out["relative_volume"] = out["volume"] / out["volume_sma20"].replace(0, np.nan)
    out["session_high"] = out["high"].groupby(session).cummax()
    out["session_low"] = out["low"].groupby(session).cummin()
    out["distance_from_vwap_pct"] = (out["close"] - out["vwap"]) / out["vwap"].replace(0, np.nan) * 100.0

    opening_high, opening_low = _session_opening_range(out)
    out["opening_range_high"] = opening_high
    out["opening_range_low"] = opening_low
    out["opening_range_breakout"] = out["close"] > out["opening_range_high"]
    out["opening_range_breakdown"] = out["close"] < out["opening_range_low"]

    out["trend"] = np.select(
        [
            (out["ema9"] > out["ema20"]) & (out["ema20"] > out["ema50"]),
            (out["ema9"] < out["ema20"]) & (out["ema20"] < out["ema50"]),
        ],
        ["UPTREND", "DOWNTREND"],
        default="SIDEWAYS",
    )
    return out
