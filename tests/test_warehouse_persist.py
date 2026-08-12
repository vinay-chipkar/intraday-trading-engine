import json

import pytest

from intraday_engine.storage.warehouse.manifest import load_manifest, load_schema_version, save_schema_version
from intraday_engine.storage.warehouse.persist import (
    WarehousePersistError,
    WarehouseSchemaVersionError,
    _distinct_partitions,
    persist_warehouse,
)
from intraday_engine.storage.warehouse.schema import SCHEMA_VERSION, TABLE_SPECS_BY_NAME

from tests._warehouse_fixtures import build_source_db


@pytest.fixture
def source_db(tmp_path):
    path = tmp_path / "source.duckdb"
    build_source_db(str(path))
    return path


def test_persist_writes_one_partition_per_distinct_date(source_db, tmp_path):
    root = tmp_path / "warehouse"
    summary = persist_warehouse(str(source_db), root)

    assert "raw/candles/date=2026-08-11/part.parquet" in summary["written"]
    assert "raw/candles/date=2026-08-12/part.parquet" in summary["written"]
    assert summary["tables"]["candles"] == {"written": 2, "skipped": 0}


def test_persist_separates_raw_research_ml_zones(source_db, tmp_path):
    root = tmp_path / "warehouse"
    persist_warehouse(str(source_db), root)

    assert (root / "raw" / "candles").exists()
    assert (root / "raw" / "instrument_master").exists()
    assert (root / "research" / "candidate_events").exists()
    # nothing was written for ml/ since feature_snapshots/training_labels are empty
    assert not (root / "ml").exists()


def test_persist_writes_dimension_table_as_single_file(source_db, tmp_path):
    root = tmp_path / "warehouse"
    persist_warehouse(str(source_db), root)
    assert (root / "raw" / "instrument_master" / "dimension" / "part.parquet").exists()


def test_persist_manifest_has_correct_row_counts_and_valid_checksums(source_db, tmp_path):
    root = tmp_path / "warehouse"
    persist_warehouse(str(source_db), root)
    manifest = load_manifest(root)

    entry = manifest["raw/candles/date=2026-08-11/part.parquet"]
    assert entry["rows"] == 2  # two AAA candles on 2026-08-11
    assert entry["table"] == "candles"
    assert entry["zone"] == "raw"

    from intraday_engine.storage.warehouse.manifest import sha256_of_file
    actual_file = root / "raw/candles/date=2026-08-11/part.parquet"
    assert sha256_of_file(actual_file) == entry["sha256"]


def test_persist_writes_no_file_for_empty_tables(source_db, tmp_path):
    root = tmp_path / "warehouse"
    persist_warehouse(str(source_db), root)
    manifest = load_manifest(root)

    for table in ("feature_snapshots", "training_labels", "market_context", "signals"):
        assert not any(entry["table"] == table for entry in manifest.values())
    assert not (root / "ml" / "feature_snapshots").exists()


def test_repeated_persist_is_idempotent_and_skips_unchanged_partitions(source_db, tmp_path):
    root = tmp_path / "warehouse"
    first = persist_warehouse(str(source_db), root)
    assert len(first["written"]) > 0

    manifest_before = load_manifest(root)
    checksums_before = {path: entry["sha256"] for path, entry in manifest_before.items()}

    second = persist_warehouse(str(source_db), root)

    assert second["written"] == []
    assert set(second["skipped"]) == set(manifest_before.keys())
    manifest_after = load_manifest(root)
    assert manifest_after == manifest_before
    assert {path: entry["sha256"] for path, entry in manifest_after.items()} == checksums_before


def test_persist_only_rewrites_the_partition_that_actually_changed(source_db, tmp_path):
    root = tmp_path / "warehouse"
    persist_warehouse(str(source_db), root)
    manifest_before = load_manifest(root)
    day1_checksum_before = manifest_before["raw/candles/date=2026-08-11/part.parquet"]["sha256"]

    # Add a new candle on 2026-08-11 (day 1 changes) but leave day 2 alone.
    import duckdb
    connection = duckdb.connect(str(source_db))
    connection.execute(
        "INSERT INTO candles VALUES "
        "('NSE_EQ|AAA','AAA', TIMESTAMPTZ '2026-08-11 03:47:00+00','1m',101,102,100,101.5,900,NULL)"
    )
    connection.close()

    summary = persist_warehouse(str(source_db), root)
    assert "raw/candles/date=2026-08-11/part.parquet" in summary["written"]
    assert "raw/candles/date=2026-08-12/part.parquet" in summary["skipped"]

    manifest_after = load_manifest(root)
    assert manifest_after["raw/candles/date=2026-08-11/part.parquet"]["sha256"] != day1_checksum_before
    assert manifest_after["raw/candles/date=2026-08-11/part.parquet"]["rows"] == 3
    assert manifest_after["raw/candles/date=2026-08-12/part.parquet"]["rows"] == 2  # unchanged


def test_persist_refuses_to_write_into_a_different_schema_version_warehouse(source_db, tmp_path):
    root = tmp_path / "warehouse"
    root.mkdir()
    save_schema_version(root, SCHEMA_VERSION + 1)

    with pytest.raises(WarehouseSchemaVersionError):
        persist_warehouse(str(source_db), root)


def test_persist_records_current_schema_version(source_db, tmp_path):
    root = tmp_path / "warehouse"
    persist_warehouse(str(source_db), root)
    assert load_schema_version(root) == {"schema_version": SCHEMA_VERSION}


def test_persist_fails_loudly_and_leaves_no_trace_when_written_rows_mismatch(source_db, tmp_path, monkeypatch):
    root = tmp_path / "warehouse"

    import intraday_engine.storage.warehouse.persist as persist_module

    real_distinct_partitions = persist_module._distinct_partitions

    def lying_distinct_partitions(connection, spec):
        values = real_distinct_partitions(connection, spec)
        if spec.name == "candles":
            # Claim one extra row exists so the post-write verification fails.
            return [(value, count + 1) for value, count in values]
        return values

    monkeypatch.setattr(persist_module, "_distinct_partitions", lying_distinct_partitions)

    with pytest.raises(WarehousePersistError, match="candles"):
        persist_warehouse(str(source_db), root)

    # No partial/temp file left behind, and nothing recorded in the manifest.
    assert not any(root.rglob("*.tmp"))
    manifest = load_manifest(root)
    assert not any(entry["table"] == "candles" for entry in manifest.values())


def test_distinct_partitions_treats_a_never_created_table_as_empty(source_db):
    import duckdb
    connection = duckdb.connect(str(source_db), read_only=True)
    try:
        # paper_observations is only created by its own ensure_observation_table(),
        # never touched by this fixture.
        spec = TABLE_SPECS_BY_NAME["paper_observations"]
        assert _distinct_partitions(connection, spec) == []
    finally:
        connection.close()
