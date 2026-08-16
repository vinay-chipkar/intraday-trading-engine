"""v4 -> v5 warehouse compatibility.

The scheduled paper-research workflow keeps a real v4 warehouse artifact
around in production (from before instrument_master_history existed). This
code must:

1. Restore that existing v4 warehouse safely with v5 code
   (instrument_master_history simply starts empty -- it never had a
   partition to restore).
2. On the next successful persist into that same warehouse, upgrade its
   on-disk schema_version to v5.
3. Continue to reject a warehouse whose schema_version is older than
   MIN_COMPATIBLE_SCHEMA_VERSION or newer than SCHEMA_VERSION.

These tests build a genuine v4-style warehouse (by persisting with the v4
TABLE_SPECS/SCHEMA_VERSION, exactly as the old code would have) rather than
hand-editing a v5 warehouse's version file, so they exercise the real
absence of any instrument_master_history partition/manifest entry.
"""

from __future__ import annotations

import intraday_engine.storage.warehouse.persist as persist_module
import pytest

from intraday_engine.storage.db import conn
from intraday_engine.storage.warehouse.manifest import load_schema_version, save_schema_version
from intraday_engine.storage.warehouse.persist import WarehouseSchemaVersionError, persist_warehouse
from intraday_engine.storage.warehouse.restore import WarehouseRestoreError, restore_warehouse
from intraday_engine.storage.warehouse.schema import (
    MIN_COMPATIBLE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TABLE_SPECS,
)

from tests._warehouse_fixtures import build_source_db


@pytest.fixture
def v4_warehouse(tmp_path):
    """A warehouse persisted exactly as v4 code would have: no
    instrument_master_history in TABLE_SPECS, stamped schema_version=4.

    The patch is scoped to just the persist call below (via
    monkeypatch.context(), undone before this fixture returns) -- the test
    body itself must run against the real, current v5 persist/restore code,
    not a patched version, or it wouldn't be testing v5 compatibility at all.
    """
    source = tmp_path / "source.duckdb"
    build_source_db(str(source))

    v4_table_specs = tuple(spec for spec in TABLE_SPECS if spec.name != "instrument_master_history")
    root = tmp_path / "warehouse"
    with pytest.MonkeyPatch.context() as m:
        m.setattr(persist_module, "TABLE_SPECS", v4_table_specs)
        m.setattr(persist_module, "SCHEMA_VERSION", 4)
        persist_warehouse(str(source), root)
    return source, root


def test_v4_warehouse_has_no_instrument_master_history_partition(v4_warehouse):
    _, root = v4_warehouse
    assert load_schema_version(root) == {"schema_version": 4}
    assert not (root / "raw" / "instrument_master_history").exists()


def test_v5_code_restores_an_existing_v4_warehouse(v4_warehouse, tmp_path):
    _, root = v4_warehouse
    target = tmp_path / "restored.duckdb"

    counts = restore_warehouse(root, str(target))

    assert counts["candles"] == 4
    assert counts["instrument_master"] == 2
    assert counts["candidate_events"] == 2
    assert counts["instrument_master_history"] == 0

    connection = conn(path=str(target))
    try:
        rows = connection.execute("SELECT COUNT(*) FROM instrument_master_history").fetchone()[0]
    finally:
        connection.close()
    assert rows == 0


def test_persisting_into_a_v4_warehouse_upgrades_it_to_v5(v4_warehouse):
    source, root = v4_warehouse
    assert load_schema_version(root) == {"schema_version": 4}

    # A new instrument_key resolution happens between the v4 warehouse's last
    # persist and now -- this row exists only in the source DB, never
    # captured under the old (v4) TABLE_SPECS.
    connection = conn(path=str(source))
    try:
        connection.execute(
            "INSERT INTO instrument_master_history VALUES "
            "('CCC', 'NSE_EQ|CCC', 'CCC Ltd', 'CCC', TIMESTAMPTZ '2026-08-13 00:00:00+00')"
        )
    finally:
        connection.close()

    summary = persist_warehouse(str(source), root)

    assert load_schema_version(root) == {"schema_version": SCHEMA_VERSION}
    assert summary["tables"]["instrument_master_history"]["written"] == 1


def test_v5_restore_after_the_upgrade_persist_matches_existing_data_exactly(v4_warehouse, tmp_path):
    source, root = v4_warehouse

    # v4 restore (what the scheduled workflow's morning job would have done
    # before this code shipped).
    pre_upgrade_target = tmp_path / "pre_upgrade.duckdb"
    restore_warehouse(root, str(pre_upgrade_target))

    # The subsequent v5 persist upgrades the warehouse in place.
    persist_warehouse(str(source), root)
    assert load_schema_version(root) == {"schema_version": SCHEMA_VERSION}

    # A later v5 restore must still succeed and every row/value already
    # present before the upgrade must be identical -- the upgrade must not
    # have altered any pre-existing data.
    post_upgrade_target = tmp_path / "post_upgrade.duckdb"
    counts = restore_warehouse(root, str(post_upgrade_target))
    assert counts["candles"] == 4
    assert counts["instrument_master"] == 2
    assert counts["candidate_events"] == 2

    pre_conn = conn(path=str(pre_upgrade_target))
    post_conn = conn(path=str(post_upgrade_target))
    try:
        for table in ("candles", "instrument_master", "candidate_events"):
            pre_rows = pre_conn.execute(f"SELECT * FROM {table} ORDER BY ALL").df()
            post_rows = post_conn.execute(f"SELECT * FROM {table} ORDER BY ALL").df()
            assert pre_rows.equals(post_rows), f"{table} changed across the v4 -> v5 upgrade"
    finally:
        pre_conn.close()
        post_conn.close()


def test_restore_rejects_a_warehouse_older_than_the_compatibility_floor(tmp_path):
    root = tmp_path / "too_old_warehouse"
    root.mkdir()
    save_schema_version(root, MIN_COMPATIBLE_SCHEMA_VERSION - 1)
    from intraday_engine.storage.warehouse.manifest import save_manifest
    save_manifest(root, {})

    with pytest.raises(WarehouseRestoreError, match="schema_version"):
        restore_warehouse(root, str(tmp_path / "target.duckdb"))


def test_persist_rejects_a_warehouse_older_than_the_compatibility_floor(v4_warehouse):
    source, root = v4_warehouse
    save_schema_version(root, MIN_COMPATIBLE_SCHEMA_VERSION - 1)

    with pytest.raises(WarehouseSchemaVersionError, match="schema_version"):
        persist_warehouse(str(source), root)


def test_restore_still_rejects_a_warehouse_newer_than_the_code(v4_warehouse, tmp_path):
    _, root = v4_warehouse
    save_schema_version(root, SCHEMA_VERSION + 1)

    with pytest.raises(WarehouseRestoreError, match="schema_version"):
        restore_warehouse(root, str(tmp_path / "target.duckdb"))
