"""ingest_symbols must leave a durable operational record behind: one row in
ingestion_runs per batch (even a total failure) and one row per symbol in
data_quality_events whenever quality_report() flags an issue. Both tables
existed in the schema from the start but nothing ever wrote to them until
this wiring -- research/monitoring.py depends on ingestion_runs history to
detect a sustained-failure pattern across days.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import intraday_engine.market.ingestion as ingestion
import intraday_engine.storage.db as db
from config.settings import settings as real_settings


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "test.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)
    connection = db.conn()
    connection.execute(
        "INSERT INTO instrument_master VALUES "
        "('AAA', 'NSE_EQ|AAA', 'AAA Ltd', 'AAA', now()),"
        "('BBB', 'NSE_EQ|BBB', 'BBB Ltd', 'BBB', now())"
    )
    connection.close()
    return fake_settings.duckdb_path


def _candle_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
    )


class _FakeAPI:
    def __init__(self, frames: dict[str, pd.DataFrame | Exception]):
        self.frames = frames

    def intraday_candles(self, instrument_key: str, unit: str = "minutes", interval: int = 1):
        frame = self.frames[instrument_key]
        if isinstance(frame, Exception):
            raise frame
        return frame


def _clean_frame(symbol: str) -> pd.DataFrame:
    return _candle_frame(
        [
            (pd.Timestamp("2026-08-12 09:15:00", tz="Asia/Kolkata"), 100, 101, 99, 100.5, 1000, 0),
            (pd.Timestamp("2026-08-12 09:16:00", tz="Asia/Kolkata"), 100.5, 102, 99, 101, 1100, 0),
        ]
    )


def test_successful_batch_writes_a_success_ingestion_run(isolated_db, monkeypatch):
    api = _FakeAPI({"NSE_EQ|AAA": _clean_frame("AAA"), "NSE_EQ|BBB": _clean_frame("BBB")})
    monkeypatch.setattr(ingestion, "UpstoxREST", lambda: api)

    results = ingestion.ingest_symbols(interval=1)
    assert len(results) == 2 and all(r.error is None for r in results)

    connection = db.conn()
    row = connection.execute(
        "SELECT requested_symbols, successful_symbols, status, error FROM ingestion_runs"
    ).fetchone()
    connection.close()
    assert row == (2, 2, "SUCCESS", None)


def test_total_failure_still_writes_a_failed_ingestion_run(isolated_db, monkeypatch):
    api = _FakeAPI(
        {"NSE_EQ|AAA": RuntimeError("401 Unauthorized"), "NSE_EQ|BBB": RuntimeError("401 Unauthorized")}
    )
    monkeypatch.setattr(ingestion, "UpstoxREST", lambda: api)

    results = ingestion.ingest_symbols(interval=1)
    with pytest.raises(ingestion.IngestionFailure):
        ingestion.assess_ingestion_results(results)

    connection = db.conn()
    row = connection.execute(
        "SELECT requested_symbols, successful_symbols, status FROM ingestion_runs"
    ).fetchone()
    connection.close()
    assert row == (2, 0, "FAILED")


def test_partial_failure_is_recorded_as_partial(isolated_db, monkeypatch):
    api = _FakeAPI({"NSE_EQ|AAA": _clean_frame("AAA"), "NSE_EQ|BBB": RuntimeError("timeout")})
    monkeypatch.setattr(ingestion, "UpstoxREST", lambda: api)

    ingestion.ingest_symbols(interval=1)

    connection = db.conn()
    status, error = connection.execute("SELECT status, error FROM ingestion_runs").fetchone()
    connection.close()
    assert status == "PARTIAL"
    assert "BBB" in error and "timeout" in error


def test_duplicate_timestamps_are_recorded_as_a_data_quality_event(isolated_db, monkeypatch):
    dupe_frame = _candle_frame(
        [
            (pd.Timestamp("2026-08-12 09:15:00", tz="Asia/Kolkata"), 100, 101, 99, 100.5, 1000, 0),
            (pd.Timestamp("2026-08-12 09:15:00", tz="Asia/Kolkata"), 100, 101, 99, 100.5, 1000, 0),
        ]
    )
    api = _FakeAPI({"NSE_EQ|AAA": dupe_frame, "NSE_EQ|BBB": _clean_frame("BBB")})
    monkeypatch.setattr(ingestion, "UpstoxREST", lambda: api)

    ingestion.ingest_symbols(interval=1)

    connection = db.conn()
    rows = connection.execute("SELECT symbol, issue_type, details FROM data_quality_events").fetchall()
    connection.close()
    assert len(rows) == 1
    assert rows[0][0] == "AAA"
    assert rows[0][1] == "INGESTION_QUALITY"
    assert "duplicates" in rows[0][2]


def test_clean_batch_writes_no_data_quality_events(isolated_db, monkeypatch):
    api = _FakeAPI({"NSE_EQ|AAA": _clean_frame("AAA"), "NSE_EQ|BBB": _clean_frame("BBB")})
    monkeypatch.setattr(ingestion, "UpstoxREST", lambda: api)

    ingestion.ingest_symbols(interval=1)

    connection = db.conn()
    count = connection.execute("SELECT COUNT(*) FROM data_quality_events").fetchone()[0]
    connection.close()
    assert count == 0


def test_empty_results_batch_is_recorded_as_failed(isolated_db, monkeypatch):
    # ingest_symbols itself always has >=1 instrument here (guarded by
    # LookupError above), but _record_ingestion_run must not divide by zero
    # or otherwise crash if it is ever handed an empty result list directly.
    ingestion._record_ingestion_run(
        started_at=pd.Timestamp.now(tz="UTC"), interval="1m", requested=0, results=[]
    )
    connection = db.conn()
    status, requested = connection.execute("SELECT status, requested_symbols FROM ingestion_runs").fetchone()
    connection.close()
    assert status == "FAILED"
    assert requested == 0
