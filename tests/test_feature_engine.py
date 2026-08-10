import numpy as np
import pandas as pd

from intraday_engine.technical.feature_engine import add_feature_engine, latest_feature_snapshot


def _ohlcv(rows=80):
    timestamps = pd.date_range(
        "2026-01-02 09:15", periods=rows, freq="min", tz="Asia/Kolkata"
    )
    close = np.linspace(100, 119, rows)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.arange(1, rows + 1) * 1000,
        "open_interest": 0,
    })


def test_feature_engine_includes_candlestick_features():
    df = _ohlcv()
    df.loc[df.index[-1], ["open", "high", "low", "close"]] = [119.0, 120.0, 113.0, 119.8]
    enriched = add_feature_engine(df)
    assert "candle_pattern" in enriched.columns
    assert "pattern_flags" in enriched.columns
    assert enriched.iloc[-1]["candle_pattern"] == "HAMMER"


def test_latest_snapshot_contains_pattern_and_flags():
    df = _ohlcv()
    df.loc[df.index[-1], ["open", "high", "low", "close"]] = [119.0, 120.0, 113.0, 119.8]
    snapshot = latest_feature_snapshot(
        df,
        symbol="TEST",
        instrument_key="NSE_EQ|TEST",
    )
    assert snapshot["candle_pattern"] == "HAMMER"
    assert '"hammer":true' in snapshot["feature_json"]
