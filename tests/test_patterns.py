import pandas as pd
import pytest

from intraday_engine.patterns.candles import add_candle_patterns, classify_candle, latest_pattern_names


def test_hammer_classification():
    row = pd.Series({"open": 100, "high": 101, "low": 95, "close": 100.8})
    assert classify_candle(row) == "HAMMER"


def test_bullish_engulfing():
    df = pd.DataFrame([
        {"open": 102, "high": 103, "low": 99, "close": 100},
        {"open": 99, "high": 104, "low": 98, "close": 103},
    ])
    out = add_candle_patterns(df)
    assert out.loc[1, "bullish_engulfing"]
    assert out.loc[1, "candle_pattern"] == "BULLISH_ENGULFING"


def test_morning_star_and_evening_star():
    morning = pd.DataFrame([
        {"open": 105, "high": 106, "low": 99, "close": 100},
        {"open": 99.5, "high": 100.0, "low": 99.2, "close": 99.8},
        {"open": 100, "high": 105, "low": 99.5, "close": 104},
    ])
    evening = pd.DataFrame([
        {"open": 100, "high": 106, "low": 99, "close": 105},
        {"open": 105.5, "high": 106.0, "low": 105.2, "close": 105.7},
        {"open": 105, "high": 105.5, "low": 99, "close": 101},
    ])
    assert add_candle_patterns(morning).loc[2, "morning_star"]
    assert add_candle_patterns(evening).loc[2, "evening_star"]


def test_inside_and_outside_bar():
    df = pd.DataFrame([
        {"open": 100, "high": 105, "low": 95, "close": 102},
        {"open": 101, "high": 104, "low": 97, "close": 103},
        {"open": 103, "high": 106, "low": 94, "close": 100},
    ])
    out = add_candle_patterns(df)
    assert out.loc[1, "inside_bar"]
    assert out.loc[2, "outside_bar"]


def test_three_soldiers_and_crows():
    soldiers = pd.DataFrame([
        {"open": 100, "high": 102.2, "low": 99.8, "close": 102},
        {"open": 102, "high": 104.2, "low": 101.8, "close": 104},
        {"open": 104, "high": 106.2, "low": 103.8, "close": 106},
    ])
    crows = pd.DataFrame([
        {"open": 106, "high": 106.2, "low": 103.8, "close": 104},
        {"open": 104, "high": 104.2, "low": 101.8, "close": 102},
        {"open": 102, "high": 102.2, "low": 99.8, "close": 100},
    ])
    assert add_candle_patterns(soldiers).loc[2, "three_white_soldiers"]
    assert add_candle_patterns(crows).loc[2, "three_black_crows"]


def test_latest_pattern_names_is_deterministic():
    df = pd.DataFrame([
        {"open": 102, "high": 103, "low": 99, "close": 100},
        {"open": 99, "high": 104, "low": 98, "close": 103},
    ])
    assert latest_pattern_names(df) == ["bullish_engulfing", "outside_bar"]


def test_missing_ohlc_is_rejected():
    with pytest.raises(ValueError, match="Missing OHLC columns"):
        add_candle_patterns(pd.DataFrame([{"open": 1, "close": 2}]))


def test_patterns_do_not_use_future_candles():
    first_two = pd.DataFrame([
        {"open": 102, "high": 103, "low": 99, "close": 100},
        {"open": 99, "high": 104, "low": 98, "close": 103},
    ])
    with_future = pd.concat([
        first_two,
        pd.DataFrame([{"open": 103, "high": 104, "low": 102, "close": 102.5}]),
    ], ignore_index=True)
    base = add_candle_patterns(first_two)
    extended = add_candle_patterns(with_future)
    assert base.loc[0, "candle_pattern"] == extended.loc[0, "candle_pattern"]
    assert base.loc[1, "candle_pattern"] == extended.loc[1, "candle_pattern"]
