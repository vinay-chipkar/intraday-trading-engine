"""Regression tests for additive warehouse schema-version compatibility."""

from __future__ import annotations

import duckdb
import pytest

from intraday_engine.storage.db import conn
from intraday_engine.storage.warehouse.manifest import load_manifest, load_schema_version, save_schema_version
from intraday_engine.storage.warehouse.persist import persist_warehouse
from intraday_engine.storage.warehouse.restore import WarehouseRestoreError, restore_warehouse
from intraday_engine.storage.warehouse.schema import SCHEMA_VERSION, TABLE_SPECS
from tests._warehouse_fixtures import build_source_db


@pytest.fixture
def source_db(tmp_path):
    path = tmp_path / "source.duckdb"
    build_source_db(str(path))
    return path


def _table_counts(db_path: str) -> dict[str, int]:
    connection = conn(path=db_path)
    try:
        return {
            spec.name: connection.execute(f"SELECT COUNT(*) FROM {spec.name}").fetchone()[0]
            for spec in TABLE_SPECS
        }
    finally:
        connection.close()


def _candle_rows(db_path: str) -> list[tuple]:
    connection = duckdb.connect(db_path, read_only=True)
    try:
        return connection.execute(
            """
            SELECT instrument_key, symbol, timestamp, interval, open, high, low, close, volume, open_interest
            FROM candles
            ORDER BY instrument_key, timestamp, interval
            """
        ).fetchall()
    finally:
        connection.close()


def test_genuine_v4_warehouse_restores_persists_as_v5_and_restores_again(source_db, tmp_path):
    """A production-style v4 artifact must survive the v5 deploy unchanged.

    We first create the warehouse with the same current data layout, then stamp
    it as v4. The v5-only instrument_master_history table has no rows in the
    source, so no v5-only partition exists -- matching the real v4 artifact.
    """
    root = tmp_path / "warehouse"
    original_counts = _table_counts(str(source_db))
    original_candles = _candle_rows(str(source_db))

    # Build a genuine v4-shaped artifact: v4 metadata and no v5-only table
    # partition. The source fixture has an empty instrument_master_history,
    # so persist naturally emits no partition for that table.
    persist_warehouse(str(source_db), root)
    manifest = load_manifest(root)
    assert not any(entry["table"] == "instrument_master_history" for entry in manifest.values())
    save_schema_version(root, 4)
    assert load_schema_version(root)["schema_version"] == 4

    # The v5 code must accept the existing v4 artifact and create the new
    # table structurally, but it must not invent historical mappings.
    restored_v4 = tmp_path / "restored_v4.duckdb"
    restore_warehouse(root, str(restored_v4))
    assert _table_counts(str(restored_v4)) == original_counts | {"instrument_master_history": 0}

    restored_connection = conn(path=str(restored_v4))
    try:
        assert restored_connection.execute("SELECT COUNT(*) FROM instrument_master_history").fetchone()[0] == 0
    finally:
        restored_connection.close()
    assert _candle_rows(str(restored_v4)) == original_candles

    # This is the scheduled workflow's next step: persist the successfully
    # restored database back into the same warehouse. Only a successful full
    # persist upgrades the metadata from v4 to v5.
    persist_warehouse(str(restored_v4), root)
    assert load_schema_version(root)["schema_version"] == SCHEMA_VERSION == 5

    # A later scheduled run now restores the upgraded v5 artifact normally,
    # with all pre-existing data unchanged and the new history table still
    # empty because no mappings were observed.
    restored_v5 = tmp_path / "restored_v5.duckdb"
    restore_warehouse(root, str(restored_v5))
    assert _table_counts(str(restored_v5)) == original_counts | {"instrument_master_history": 0}
    assert _candle_rows(str(restored_v5)) == original_candles

    connection = conn(path=str(restored_v5))
    try:
        assert connection.execute("SELECT COUNT(*) FROM instrument_master_history").fetchone()[0] == 0
    finally:
        connection.close()


@pytest.mark.parametrize("incompatible_version", [3, 6, "5", None])
def test_restore_rejects_unknown_or_incompatible_schema_versions(source_db, tmp_path, incompatible_version):
    root = tmp_path / f"warehouse-{str(incompatible_version).replace('/', '_')}"
    persist_warehouse(str(source_db), root)
    save_schema_version(root, incompatible_version)

    with pytest.raises(WarehouseRestoreError, match="outside the range"):
        restore_warehouse(root, str(tmp_path / "target.duckdb"))
