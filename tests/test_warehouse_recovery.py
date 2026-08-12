"""Tests for the scenarios Phase 4 adds on top of Phase 1-3's warehouse:
idempotent repeated runs, partial failure + retry, and recovery of a
multi-day warehouse chain after a simulated failed workflow run.
"""

from __future__ import annotations

import duckdb
import pytest

import intraday_engine.storage.warehouse.persist as persist_module
from intraday_engine.storage.db import conn
from intraday_engine.storage.warehouse.manifest import load_manifest
from intraday_engine.storage.warehouse.persist import WarehousePersistError, persist_warehouse
from intraday_engine.storage.warehouse.restore import restore_warehouse

from tests._warehouse_fixtures import build_source_db


def _add_candle(db_path: str, symbol: str, ts: str, price: float) -> None:
    connection = duckdb.connect(db_path)
    connection.execute(
        f"INSERT INTO candles VALUES (?, ?, TIMESTAMPTZ '{ts}', '1m', ?, ?, ?, ?, ?, NULL)",
        [f"NSE_EQ|{symbol}", symbol, price, price + 1, price - 1, price + 0.5, 1000],
    )
    connection.close()


def _candle_count(db_path: str) -> int:
    connection = duckdb.connect(db_path, read_only=True)
    try:
        return connection.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    finally:
        connection.close()


@pytest.fixture
def source_db(tmp_path):
    path = tmp_path / "source.duckdb"
    build_source_db(str(path))
    return path


# --- idempotent repeated runs (full persist -> restore -> persist -> restore) ---


def test_repeated_full_cycle_never_duplicates_data(source_db, tmp_path):
    root = tmp_path / "warehouse"

    persist_warehouse(str(source_db), root)
    restore_warehouse(root, str(tmp_path / "r1.duckdb"))
    persist_warehouse(str(source_db), root)  # nothing changed -- must be a full no-op
    restore_warehouse(root, str(tmp_path / "r2.duckdb"))

    assert _candle_count(str(tmp_path / "r1.duckdb")) == _candle_count(str(tmp_path / "r2.duckdb"))
    assert _candle_count(str(tmp_path / "r2.duckdb")) == _candle_count(str(source_db))


# --- partial failure mid-persist, then retry ---


def test_partial_failure_leaves_completed_partitions_valid(source_db, tmp_path, monkeypatch):
    root = tmp_path / "warehouse"
    real_distinct_partitions = persist_module._distinct_partitions

    def crash_on_candidate_events(connection, spec):
        if spec.name == "candidate_events":
            raise RuntimeError("simulated crash mid-run (e.g. runner killed)")
        return real_distinct_partitions(connection, spec)

    monkeypatch.setattr(persist_module, "_distinct_partitions", crash_on_candidate_events)

    with pytest.raises(RuntimeError, match="simulated crash"):
        persist_warehouse(str(source_db), root)

    # candles comes before candidate_events in TABLE_SPECS, so it should have
    # completed and verified successfully before the crash.
    manifest = load_manifest(root)
    assert any(entry["table"] == "candles" for entry in manifest.values())
    assert not any(entry["table"] == "candidate_events" for entry in manifest.values())
    assert not any(root.rglob("*.tmp"))


def test_retry_after_partial_failure_completes_and_matches_a_clean_run(source_db, tmp_path, monkeypatch):
    crashed_root = tmp_path / "crashed_warehouse"
    clean_root = tmp_path / "clean_warehouse"
    real_distinct_partitions = persist_module._distinct_partitions

    calls = {"n": 0}

    def crash_once_on_candidate_events(connection, spec):
        if spec.name == "candidate_events" and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("simulated crash")
        return real_distinct_partitions(connection, spec)

    monkeypatch.setattr(persist_module, "_distinct_partitions", crash_once_on_candidate_events)
    with pytest.raises(RuntimeError):
        persist_warehouse(str(source_db), crashed_root)

    monkeypatch.undo()  # restore the real function for the retry
    persist_warehouse(str(source_db), crashed_root)  # retry: picks up where it left off
    persist_warehouse(str(source_db), clean_root)  # a normal, uninterrupted run for comparison

    restore_warehouse(crashed_root, str(tmp_path / "from_crashed.duckdb"))
    restore_warehouse(clean_root, str(tmp_path / "from_clean.duckdb"))

    from intraday_engine.storage.warehouse.schema import TABLE_SPECS

    for spec in TABLE_SPECS:
        c1 = conn(path=str(tmp_path / "from_crashed.duckdb"))
        c2 = conn(path=str(tmp_path / "from_clean.duckdb"))
        try:
            n1 = c1.execute(f"SELECT COUNT(*) FROM {spec.name}").fetchone()[0]
            n2 = c2.execute(f"SELECT COUNT(*) FROM {spec.name}").fetchone()[0]
            assert n1 == n2, f"{spec.name}: retried={n1} vs clean={n2}"
        finally:
            c1.close()
            c2.close()


# --- multi-day chain: restore -> accumulate -> persist -> restore again ---


def test_multi_day_chain_accumulates_correctly_across_simulated_days(source_db, tmp_path):
    """Simulates what the real workflow does daily: download+restore the
    warehouse, add a day's new data, persist back, and confirm the next
    day's restore sees everything so far."""
    root = tmp_path / "warehouse"

    # Day 1
    persist_warehouse(str(source_db), root)

    # Day 2: restore into a fresh DB (as the morning job would), add new data,
    # persist back to the SAME warehouse root.
    day2_db = tmp_path / "day2.duckdb"
    restore_warehouse(root, str(day2_db))
    _add_candle(str(day2_db), "AAA", "2026-08-13 03:45:00+00", 200.0)
    persist_warehouse(str(day2_db), root)

    # Day 3: restore again and confirm day 1 + day 2 data are both present.
    day3_db = tmp_path / "day3.duckdb"
    restore_warehouse(root, str(day3_db))

    assert _candle_count(str(day3_db)) == _candle_count(str(day2_db))
    connection = conn(path=str(day3_db))
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM candles WHERE timestamp = TIMESTAMPTZ '2026-08-13 03:45:00+00'"
        ).fetchone()[0] == 1
        # day 1's original dates are still there, untouched
        assert connection.execute(
            "SELECT COUNT(*) FROM candles WHERE timestamp::date = DATE '2026-08-11'"
        ).fetchone()[0] == 2
    finally:
        connection.close()


def test_recovery_after_a_failed_persist_only_loses_that_days_unwritten_increment(source_db, tmp_path, monkeypatch):
    """A crash partway through persisting "day 2" must not corrupt or lose
    "day 1"'s already-durable data -- the warehouse degrades gracefully to
    "day 1 plus whatever of day 2 finished," never to something unusable."""
    root = tmp_path / "warehouse"
    persist_warehouse(str(source_db), root)  # day 1, fully durable

    day2_db = tmp_path / "day2.duckdb"
    restore_warehouse(root, str(day2_db))
    _add_candle(str(day2_db), "AAA", "2026-08-13 03:45:00+00", 200.0)

    real_distinct_partitions = persist_module._distinct_partitions

    def crash_on_instrument_master(connection, spec):
        if spec.name == "instrument_master":
            raise RuntimeError("simulated runner failure")
        return real_distinct_partitions(connection, spec)

    monkeypatch.setattr(persist_module, "_distinct_partitions", crash_on_instrument_master)
    with pytest.raises(RuntimeError):
        persist_warehouse(str(day2_db), root)
    monkeypatch.undo()

    # The warehouse must still be fully restorable despite the interrupted run.
    recovered_db = tmp_path / "recovered.duckdb"
    restore_warehouse(root, str(recovered_db))
    connection = conn(path=str(recovered_db))
    try:
        # candles (processed before instrument_master crashed) picked up day 2's new row
        assert connection.execute(
            "SELECT COUNT(*) FROM candles WHERE timestamp = TIMESTAMPTZ '2026-08-13 03:45:00+00'"
        ).fetchone()[0] == 1
        # original day-1 data is intact, not corrupted by the interrupted run
        assert connection.execute("SELECT COUNT(*) FROM instrument_master").fetchone()[0] == 2
    finally:
        connection.close()


# --- CLI-level fail-loud behavior ---


def test_warehouse_restore_cli_propagates_failure_as_an_exception(tmp_path, monkeypatch):
    from scripts import warehouse_restore

    monkeypatch.setattr(
        "sys.argv",
        ["warehouse_restore", "--root", str(tmp_path / "does_not_exist"), "--target", str(tmp_path / "t.duckdb")],
    )
    with pytest.raises(Exception, match="does not exist"):
        warehouse_restore.main()


def test_warehouse_persist_cli_propagates_failure_as_an_exception(source_db, tmp_path, monkeypatch):
    from intraday_engine.storage.warehouse.manifest import save_schema_version
    from intraday_engine.storage.warehouse.schema import SCHEMA_VERSION
    from scripts import warehouse_persist

    root = tmp_path / "warehouse"
    root.mkdir()
    save_schema_version(root, SCHEMA_VERSION + 1)

    monkeypatch.setattr(
        "sys.argv",
        ["warehouse_persist", "--source", str(source_db), "--root", str(root)],
    )
    with pytest.raises(Exception, match="schema_version"):
        warehouse_persist.main()
