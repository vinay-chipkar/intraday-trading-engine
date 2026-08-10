import pandas as pd
import pytest

from intraday_engine.patterns.market_structure import add_market_structure, latest_market_structure


def _frame(highs, lows, closes=None):
    if closes is None:
        closes = highs
    n = len(highs)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:15", periods=n, freq="min", tz="Asia/Kolkata"),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * n,
        }
    )


def test_pivot_is_only_exposed_after_confirmation():
    df = _frame([100, 105, 102, 110], [95, 98, 97, 100])
    out = add_market_structure(df, pivot_left=1, pivot_right=1)

    assert pd.isna(out.loc[1, "pivot_high"])
    assert out.loc[2, "pivot_high"] == pytest.approx(105.0)


def test_higher_highs_and_higher_lows_create_bullish_structure():
    df = _frame(
        [100, 102, 106, 104, 110, 108, 115, 112],
        [95, 98, 96, 100, 98, 104, 101, 107],
        [98, 101, 103, 102, 108, 106, 112, 110],
    )
    out = add_market_structure(df, pivot_left=1, pivot_right=1)
    latest = out.iloc[-1]

    assert latest["market_structure"] == "BULLISH"
    assert latest["structure_trend"] == "UPTREND"
    assert latest["support"] == pytest.approx(101.0)
    assert latest["resistance"] == pytest.approx(115.0)


def test_double_top_requires_separation_and_price_tolerance():
    df = _frame(
        [100, 110, 105, 104, 109, 103],
        [95, 98, 97, 96, 98, 97],
        [98, 108, 100, 102, 107, 101],
    )
    out = add_market_structure(
        df,
        pivot_left=1,
        pivot_right=1,
        double_pattern_tolerance=0.01,
        double_pattern_min_bars=3,
    )

    assert out.iloc[-1]["double_top"] == True


def test_latest_market_structure_is_serializable():
    df = _frame(
        [100, 102, 106, 104, 110, 108, 115, 112],
        [95, 98, 96, 100, 98, 104, 101, 107],
        [98, 101, 103, 102, 108, 106, 112, 110],
    )
    snapshot = latest_market_structure(df, pivot_left=1, pivot_right=1)

    assert snapshot["market_structure"] == "BULLISH"
    assert snapshot["structure_trend"] == "UPTREND"
    assert snapshot["support"] == pytest.approx(101.0)
    assert snapshot["resistance"] == pytest.approx(115.0)
