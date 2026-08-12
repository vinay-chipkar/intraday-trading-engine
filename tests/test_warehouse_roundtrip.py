import pandas as pd
import pytest

from intraday_engine.storage.db import conn
from intraday_engine.storage.warehouse.persist import persist_warehouse
from intraday_engine.storage.warehouse.restore import restore_warehouse
from intraday_engine.storage.warehouse.schema import TABLE_SPECS

from tests._warehouse_fixtures import build_source_db


@pytest.fixture
def source_and_warehouse(tmp_path):
    source = tmp_path / "source.duckdb"
    build_source_db(str(source))
    root = tmp_path / "warehouse"
    persist_warehouse(str(source), root)
    return source, root


def _row_counts(db_path: str) -> dict[str, int]:
    connection = conn(path=db_path)
    try:
        return {
            spec.name: connection.execute(f"SELECT COUNT(*) FROM {spec.name}").fetchone()[0]
            for spec in TABLE_SPECS
        }
    finally:
        connection.close()


def test_round_trip_row_counts_match_for_every_table(source_and_warehouse, tmp_path):
    source, root = source_and_warehouse
    target = tmp_path / "restored.duckdb"
    restore_warehouse(root, str(target))

    assert _row_counts(str(target)) == _row_counts(str(source))


def test_round_trip_candle_data_is_row_for_row_identical(source_and_warehouse, tmp_path):
    source, root = source_and_warehouse
    target = tmp_path / "restored.duckdb"
    restore_warehouse(root, str(target))

    source_conn = conn(path=str(source))
    target_conn = conn(path=str(target))
    try:
        source_df = source_conn.execute("SELECT * FROM candles ORDER BY symbol, timestamp").df()
        restored_df = target_conn.execute("SELECT * FROM candles ORDER BY symbol, timestamp").df()
    finally:
        source_conn.close()
        target_conn.close()

    pd.testing.assert_frame_equal(source_df, restored_df)


def test_round_trip_preserves_timezone_aware_timestamps_exactly(source_and_warehouse, tmp_path):
    # The fixture inserts one candle with a +05:30 offset and others with +00 --
    # all must round-trip to the exact same instant (not shifted, not made naive).
    source, root = source_and_warehouse
    target = tmp_path / "restored.duckdb"
    restore_warehouse(root, str(target))

    target_conn = conn(path=str(target))
    try:
        rows = target_conn.execute(
            "SELECT symbol, timestamp FROM candles WHERE symbol = 'BBB'"
        ).fetchall()
    finally:
        target_conn.close()

    assert len(rows) == 1
    symbol, ts = rows[0]
    assert ts.tzinfo is not None
    # 2026-08-12 09:15:00+05:30 is 2026-08-12 03:45:00+00:00
    assert ts.astimezone(pd.Timestamp("2026-08-12", tz="UTC").tzinfo) == pd.Timestamp(
        "2026-08-12 03:45:00", tz="UTC"
    )


def test_round_trip_preserves_empty_tables_as_empty_not_missing(source_and_warehouse, tmp_path):
    source, root = source_and_warehouse
    target = tmp_path / "restored.duckdb"
    restore_warehouse(root, str(target))

    target_conn = conn(path=str(target))
    try:
        tables = {
            row[0]
            for row in target_conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
        }
        for spec in TABLE_SPECS:
            assert spec.name in tables
        assert target_conn.execute("SELECT COUNT(*) FROM feature_snapshots").fetchone()[0] == 0
        assert target_conn.execute("SELECT COUNT(*) FROM training_labels").fetchone()[0] == 0
    finally:
        target_conn.close()


def test_repeated_persist_then_restore_produces_no_duplicates(source_and_warehouse, tmp_path):
    source, root = source_and_warehouse
    # Persist again with no source changes -- must be a full no-op.
    persist_warehouse(str(source), root)
    persist_warehouse(str(source), root)

    target = tmp_path / "restored.duckdb"
    restore_warehouse(root, str(target))

    assert _row_counts(str(target)) == _row_counts(str(source))
