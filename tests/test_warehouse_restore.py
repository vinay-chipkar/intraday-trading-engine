import json

import pytest

from intraday_engine.storage.warehouse.manifest import (
    load_manifest,
    save_manifest,
    save_schema_version,
)
from intraday_engine.storage.warehouse.persist import persist_warehouse
from intraday_engine.storage.warehouse.restore import WarehouseRestoreError, restore_warehouse
from intraday_engine.storage.warehouse.schema import SCHEMA_VERSION

from tests._warehouse_fixtures import build_source_db


@pytest.fixture
def warehouse(tmp_path):
    source = tmp_path / "source.duckdb"
    build_source_db(str(source))
    root = tmp_path / "warehouse"
    persist_warehouse(str(source), root)
    return root


def test_restore_raises_when_warehouse_root_is_missing(tmp_path):
    with pytest.raises(WarehouseRestoreError, match="does not exist"):
        restore_warehouse(tmp_path / "nope", str(tmp_path / "target.duckdb"))


def test_restore_raises_when_schema_version_file_is_missing(tmp_path):
    root = tmp_path / "warehouse"
    root.mkdir()
    with pytest.raises(WarehouseRestoreError, match="_schema_version.json"):
        restore_warehouse(root, str(tmp_path / "target.duckdb"))


def test_restore_raises_on_schema_version_mismatch(warehouse, tmp_path):
    save_schema_version(warehouse, SCHEMA_VERSION + 1)
    with pytest.raises(WarehouseRestoreError, match="schema_version"):
        restore_warehouse(warehouse, str(tmp_path / "target.duckdb"))


def test_restore_raises_when_a_manifest_partition_file_is_missing(warehouse, tmp_path):
    candles_file = warehouse / "raw/candles/date=2026-08-11/part.parquet"
    candles_file.unlink()

    with pytest.raises(WarehouseRestoreError, match="missing partition file"):
        restore_warehouse(warehouse, str(tmp_path / "target.duckdb"))


def test_restore_raises_on_checksum_mismatch_from_corrupted_file(warehouse, tmp_path):
    candles_file = warehouse / "raw/candles/date=2026-08-11/part.parquet"
    original = candles_file.read_bytes()
    candles_file.write_bytes(original[:-20] + b"\x00" * 20)  # corrupt the tail

    with pytest.raises(WarehouseRestoreError, match="checksum mismatch"):
        restore_warehouse(warehouse, str(tmp_path / "target.duckdb"))


def test_restore_raises_on_row_count_mismatch_against_manifest(warehouse, tmp_path):
    manifest = load_manifest(warehouse)
    manifest["raw/candles/date=2026-08-11/part.parquet"]["rows"] = 999
    save_manifest(warehouse, manifest)

    with pytest.raises(WarehouseRestoreError, match="candles"):
        restore_warehouse(warehouse, str(tmp_path / "target.duckdb"))


def test_restore_succeeds_and_reports_row_counts(warehouse, tmp_path):
    target = tmp_path / "target.duckdb"
    counts = restore_warehouse(warehouse, str(target))

    assert counts["candles"] == 4
    assert counts["instrument_master"] == 2
    assert counts["candidate_events"] == 2
    assert counts["feature_snapshots"] == 0
    assert counts["training_labels"] == 0


def test_restored_database_has_identical_schema_to_a_fresh_one(warehouse, tmp_path):
    from intraday_engine.storage.db import conn

    target = tmp_path / "target.duckdb"
    restore_warehouse(warehouse, str(target))

    fresh = tmp_path / "fresh.duckdb"
    conn(path=str(fresh)).close()

    restored_conn = conn(path=str(target))
    fresh_conn = conn(path=str(fresh))
    try:
        for table in ("candles", "instrument_master", "candidate_events", "feature_snapshots"):
            restored_cols = restored_conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position", [table]
            ).fetchall()
            fresh_cols = fresh_conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position", [table]
            ).fetchall()
            assert restored_cols == fresh_cols
    finally:
        restored_conn.close()
        fresh_conn.close()


def test_restore_target_is_overwritten_if_it_already_exists(warehouse, tmp_path):
    target = tmp_path / "target.duckdb"
    target.write_bytes(b"not a real duckdb file")

    counts = restore_warehouse(warehouse, str(target))
    assert counts["candles"] == 4


def test_restore_handles_an_empty_but_versioned_warehouse(tmp_path):
    root = tmp_path / "empty_warehouse"
    root.mkdir()
    save_schema_version(root, SCHEMA_VERSION)
    save_manifest(root, {})

    counts = restore_warehouse(root, str(tmp_path / "target.duckdb"))
    assert all(count == 0 for count in counts.values())
