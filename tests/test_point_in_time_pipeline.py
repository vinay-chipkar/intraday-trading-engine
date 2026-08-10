import pandas as pd

from intraday_engine.strategy.point_in_time import enrich_point_in_time


def _bars() -> pd.DataFrame:
    rows = []
    closes = [100, 101, 99, 102, 101, 104, 103, 106, 105, 108, 107, 110]
    for i, close in enumerate(closes):
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-08-10 09:15:00") + pd.Timedelta(minutes=i),
                "symbol": "TEST",
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100_000 + i * 1_000,
            }
        )
    return pd.DataFrame(rows)


def test_future_bars_cannot_change_existing_features():
    original = _bars()
    mutated = original.copy()
    mutated.loc[len(mutated) - 1, ["high", "low", "close"]] = [250, 50, 200]

    left = enrich_point_in_time(original, pivot_left=2, pivot_right=2)
    right = enrich_point_in_time(mutated, pivot_left=2, pivot_right=2)

    feature_columns = [
        "confirmed_pivot_high",
        "confirmed_pivot_low",
        "support",
        "resistance",
        "trend",
        "breakout",
        "breakdown",
        "rsi14",
        "atr14",
        "vwap",
    ]
    cutoff = len(original) - 1
    pd.testing.assert_frame_equal(
        left.loc[:cutoff - 1, feature_columns],
        right.loc[:cutoff - 1, feature_columns],
        check_dtype=False,
    )


def test_pivot_becomes_available_only_after_confirmation_bars():
    df = pd.DataFrame(
        [
            {"timestamp": i, "open": o, "high": h, "low": l, "close": c, "volume": 100}
            for i, (o, h, l, c) in enumerate(
                [
                    (99, 100, 98, 99),
                    (100, 103, 99, 102),
                    (102, 110, 101, 109),
                    (109, 104, 100, 102),
                    (102, 103, 99, 100),
                ]
            )
        ]
    )

    out = enrich_point_in_time(df, pivot_left=1, pivot_right=1)

    assert pd.isna(out.loc[2, "confirmed_pivot_high"])
    assert out.loc[3, "confirmed_pivot_high"] == 110
