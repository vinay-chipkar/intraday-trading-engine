import numpy as np
import pandas as pd

from intraday_engine.technical.feature_engine import latest_feature_snapshot
from intraday_engine.technical.indicators import add_indicators, adx, bollinger_bands, macd


def candles(rows=260):
    timestamps = pd.date_range(
        "2026-08-03 09:15", periods=rows, freq="min", tz="Asia/Kolkata"
    )
    close = 100 + np.cumsum(np.full(rows, 0.08)) + np.sin(np.arange(rows) / 8)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1000, 5000, rows),
        }
    )


def test_indicators_are_added_and_warmup_is_respected():
    df = add_indicators(candles())
    expected = {
        "ema9", "ema20", "ema50", "ema200", "rsi14", "atr14", "atr_pct",
        "vwap", "macd", "macd_signal", "macd_histogram", "bb_middle", "bb_upper",
        "bb_lower", "bb_width_pct", "adx14", "plus_di14", "minus_di14",
        "session_volume", "relative_volume", "session_high", "session_low",
        "opening_range_high", "opening_range_low", "trend",
    }
    assert expected.issubset(df.columns)
    assert df["ema9"].iloc[:8].isna().all()
    assert df["ema200"].iloc[:199].isna().all()
    assert df["rsi14"].between(0, 100).all()


def test_macd_and_bollinger_shapes_match_input():
    df = candles()
    macd_line, signal, histogram = macd(df["close"])
    middle, upper, lower = bollinger_bands(df["close"])
    assert len(macd_line) == len(df)
    assert len(signal) == len(df)
    assert len(histogram) == len(df)
    assert (upper.dropna() >= middle.dropna()).all()
    assert (middle.dropna() >= lower.dropna()).all()


def test_adx_components_are_non_negative():
    adx_value, plus_di, minus_di = adx(candles())
    assert (adx_value.dropna() >= 0).all()
    assert (plus_di.dropna() >= 0).all()
    assert (minus_di.dropna() >= 0).all()


def test_feature_snapshot_has_point_in_time_schema():
    snapshot = latest_feature_snapshot(
        candles(), symbol="TEST", instrument_key="NSE_EQ|TEST"
    )
    assert snapshot["symbol"] == "TEST"
    assert snapshot["instrument_key"] == "NSE_EQ|TEST"
    assert snapshot["timeframe"] == "1m"
    assert snapshot["trend"] in {"UPTREND", "DOWNTREND", "SIDEWAYS"}
    assert 0 <= snapshot["feature_score"] <= 100
    assert "adx14" in snapshot["feature_json"]
