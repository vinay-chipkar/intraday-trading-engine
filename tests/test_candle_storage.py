"""upsert_candles: repeated ingestion must reconcile revised same-timestamp
OHLCV values instead of silently keeping the stale original (Upstox commonly
revises a recent provisional bar as more trades settle)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.storage.db import insert_candles, upsert_candles


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "test.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)


def _candle_df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["instrument_key", "symbol", "timestamp", "interval", "open", "high", "low", "close", "volume", "open_interest"],
    )


def _one(ts="2026-08-13 09:15:00+00", o=100.0, h=101.0, low=99.0, c=100.5, v=1000.0):
    return ("NSE_EQ|AAA", "AAA", pd.Timestamp(ts), "1m", o, h, low, c, v, None)


def test_upsert_inserts_genuinely_new_candles(isolated_db):
    result = upsert_candles(_candle_df([_one()]))
    assert result["inserted"] == 1
    assert result["revised"] == []


def test_upsert_reconciles_a_revised_same_timestamp_candle(isolated_db):
    upsert_candles(_candle_df([_one(c=100.5, v=1000.0)]))
    # Upstox corrects the same bar: close and volume both change.
    result = upsert_candles(_candle_df([_one(c=101.2, v=1500.0)]))

    assert result["inserted"] == 0  # not a new row
    assert len(result["revised"]) == 1
    revision = result["revised"][0]
    assert revision["old_close"] == pytest.approx(100.5)
    assert revision["new_close"] == pytest.approx(101.2)
    assert revision["old_volume"] == pytest.approx(1000.0)
    assert revision["new_volume"] == pytest.approx(1500.0)

    connection = db.conn()
    stored = connection.execute(
        "SELECT close, volume FROM candles WHERE instrument_key='NSE_EQ|AAA' AND timestamp=?",
        [pd.Timestamp("2026-08-13 09:15:00+00")],
    ).fetchone()
    connection.close()
    assert stored == (pytest.approx(101.2), pytest.approx(1500.0))  # the revision was actually applied


def test_upsert_is_a_no_op_for_unchanged_candles(isolated_db):
    upsert_candles(_candle_df([_one()]))
    result = upsert_candles(_candle_df([_one()]))  # identical values, re-fetched
    assert result["inserted"] == 0
    assert result["revised"] == []  # no spurious revision record for identical data


def test_upsert_reconciles_a_bar_older_than_the_latest_stored_one(isolated_db):
    # The bug this fixes: a naive "only look at timestamp > last_stored"
    # filter would never even see a correction to an *older* already-stored
    # bar. upsert_candles must reconcile it regardless of position.
    upsert_candles(_candle_df([_one(ts="2026-08-13 09:15:00+00", c=100.0)]))
    upsert_candles(_candle_df([_one(ts="2026-08-13 09:16:00+00", c=101.0)]))  # now "latest"

    # Re-fetch of the whole day's candles reveals bar #1 (no longer the
    # latest) was corrected.
    result = upsert_candles(
        _candle_df([
            _one(ts="2026-08-13 09:15:00+00", c=100.75),  # revised, not the latest bar
            _one(ts="2026-08-13 09:16:00+00", c=101.0),   # unchanged
        ])
    )
    assert result["inserted"] == 0
    assert len(result["revised"]) == 1
    assert str(result["revised"][0]["timestamp"]) .startswith("2026-08-13 09:15:00")


def test_insert_candles_still_ignores_conflicts_for_bulk_backfill(isolated_db):
    # insert_candles (used by one-time historical backfill) intentionally
    # keeps its simpler INSERT OR IGNORE semantics -- unchanged by this fix.
    insert_candles(_candle_df([_one(c=100.5)]))
    inserted_again = insert_candles(_candle_df([_one(c=999.0)]))  # "revised" value ignored
    assert inserted_again == 0
    connection = db.conn()
    stored_close = connection.execute(
        "SELECT close FROM candles WHERE instrument_key='NSE_EQ|AAA'"
    ).fetchone()[0]
    connection.close()
    assert stored_close == pytest.approx(100.5)  # untouched, as insert_candles has always behaved
