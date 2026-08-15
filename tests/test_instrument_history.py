"""upsert_instruments: instrument_master only ever holds the CURRENT
symbol -> instrument_key mapping (keyed by symbol, overwritten on every
resolve), so a relisting/rename silently erases the previous mapping unless
it is separately logged. instrument_master_history is the append-only audit
trail that lets a historical observation be traced back to the exact
instrument mapping in effect when it was recorded."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.storage.db import instrument_history_for_symbol, upsert_instruments


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "test.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)


def _row(symbol, instrument_key, updated_at, name=None, trading_symbol=None):
    return {
        "symbol": symbol,
        "instrument_key": instrument_key,
        "name": name or symbol,
        "trading_symbol": trading_symbol or symbol,
        "updated_at": pd.Timestamp(updated_at, tz="Asia/Kolkata"),
    }


def test_relisting_preserves_old_instrument_key_in_history(isolated_db):
    upsert_instruments(pd.DataFrame([_row("ZOMATO", "NSE_EQ|OLD_KEY", "2026-01-01 09:00:00")]))
    upsert_instruments(pd.DataFrame([_row("ZOMATO", "NSE_EQ|NEW_KEY", "2026-06-01 09:00:00", name="Eternal")]))

    current = db.conn().execute(
        "SELECT instrument_key FROM instrument_master WHERE symbol = 'ZOMATO'"
    ).fetchone()
    assert current[0] == "NSE_EQ|NEW_KEY"

    history = instrument_history_for_symbol("ZOMATO")
    assert list(history["instrument_key"]) == ["NSE_EQ|OLD_KEY", "NSE_EQ|NEW_KEY"]


def test_resync_of_unchanged_mapping_does_not_duplicate_history(isolated_db):
    upsert_instruments(pd.DataFrame([_row("INFY", "NSE_EQ|INFY", "2026-01-01 09:00:00")]))
    upsert_instruments(pd.DataFrame([_row("INFY", "NSE_EQ|INFY", "2026-01-02 09:00:00")]))
    upsert_instruments(pd.DataFrame([_row("INFY", "NSE_EQ|INFY", "2026-01-03 09:00:00")]))

    history = instrument_history_for_symbol("INFY")
    assert len(history) == 1
    assert history["instrument_key"].iloc[0] == "NSE_EQ|INFY"


def test_history_is_scoped_per_symbol(isolated_db):
    upsert_instruments(pd.DataFrame([
        _row("AAA", "NSE_EQ|AAA", "2026-01-01 09:00:00"),
        _row("BBB", "NSE_EQ|BBB", "2026-01-01 09:00:00"),
    ]))

    assert list(instrument_history_for_symbol("AAA")["instrument_key"]) == ["NSE_EQ|AAA"]
    assert list(instrument_history_for_symbol("BBB")["instrument_key"]) == ["NSE_EQ|BBB"]
    assert instrument_history_for_symbol("CCC").empty
