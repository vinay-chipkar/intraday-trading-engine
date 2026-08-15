from datetime import date

import pandas as pd

from intraday_engine.market.backfill import date_windows
from intraday_engine.market.candles import normalize_candles, quality_report


def test_date_windows_cover_range_without_overlap():
    windows = list(date_windows(date(2026, 1, 1), date(2026, 2, 10), max_days=30))
    assert windows == [
        (date(2026, 1, 1), date(2026, 1, 30)),
        (date(2026, 1, 31), date(2026, 2, 10)),
    ]


def test_normalize_candles_sorts_deduplicates_and_adds_identity():
    raw = pd.DataFrame(
        [
            ["2026-08-10T03:17:00Z", 101, 103, 100, 102, 1000, 0],
            ["2026-08-10T03:16:00Z", 99, 101, 98, 100, 900, 0],
            ["2026-08-10T03:17:00Z", 101, 104, 100, 103, 1100, 0],
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"],
    )

    normalized = normalize_candles(
        raw,
        instrument_key="NSE_EQ|TEST",
        symbol="TEST",
        interval="1m",
    )

    assert len(normalized) == 2
    assert normalized["timestamp"].is_monotonic_increasing
    assert normalized.iloc[-1]["close"] == 103
    assert normalized.iloc[-1]["symbol"] == "TEST"
    assert normalized.iloc[-1]["interval"] == "1m"


def test_quality_report_detects_duplicate_and_bad_ohlc():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-10 09:15", "2026-08-10 09:15"]),
            "open": [100, 100],
            "high": [99, 101],
            "low": [98, 99],
            "close": [98.5, 100],
            "volume": [100, -1],
        }
    )
    report = quality_report(frame)
    assert report["duplicates"] == 1
    assert report["invalid_ohlc"] == 1
    assert report["negative_volume"] == 1


def test_quality_report_detects_a_missing_minute_gap():
    # 09:15, 09:16, 09:18 -- 09:17 is missing.
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-08-10 09:15", "2026-08-10 09:16", "2026-08-10 09:18"], utc=True
            ),
            "open": [100, 100, 100],
            "high": [101, 101, 101],
            "low": [99, 99, 99],
            "close": [100, 100, 100],
            "volume": [1000, 1000, 1000],
        }
    )
    report = quality_report(frame)
    assert report["session_gaps"] == 1


def test_quality_report_no_gaps_for_a_contiguous_sequence():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-08-10 09:15", "2026-08-10 09:16", "2026-08-10 09:17"], utc=True
            ),
            "open": [100, 100, 100],
            "high": [101, 101, 101],
            "low": [99, 99, 99],
            "close": [100, 100, 100],
            "volume": [1000, 1000, 1000],
        }
    )
    report = quality_report(frame)
    assert report["session_gaps"] == 0


def test_quality_report_flags_a_candle_outside_nse_session_hours():
    # 09:15 IST is well inside the session; 20:00 UTC (~01:30 IST next day
    # once converted, i.e. well outside 09:15-15:30) is not.
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-08-10 03:45:00Z", "2026-08-10 20:00:00Z"], utc=True
            ),
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
            "volume": [1000, 1000],
        }
    )
    report = quality_report(frame)
    assert report["outside_session"] == 1


def test_quality_report_new_fields_present_for_empty_frame():
    report = quality_report(pd.DataFrame())
    assert report["session_gaps"] == 0
    assert report["outside_session"] == 0
