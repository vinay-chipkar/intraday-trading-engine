"""Restore a partitioned Parquet warehouse into a fresh DuckDB.

Verification happens in two passes, both before any data is loaded:

1. Schema version: the warehouse's `_schema_version.json` must exist and match
   the code's SCHEMA_VERSION exactly.
2. Every manifest entry's file must exist and its sha256 must match what the
   manifest recorded.

Only after every partition has been verified does restore open the target
DuckDB (via the app's own schema DDL, so the restored database is
structurally identical to a freshly-migrated one) and load each table's
verified files with `read_parquet`, re-checking the aggregate row count
against the manifest one more time immediately before the INSERT.
"""

from __future__ import annotations

from pathlib import Path

from intraday_engine.storage.db import conn as open_connection
from intraday_engine.storage.warehouse.manifest import load_manifest, load_schema_version, sha256_of_file
from intraday_engine.storage.warehouse.schema import SCHEMA_VERSION, TABLE_SPECS


def _ensure_all_known_tables(target_db_path: str) -> None:
    """Create every table TABLE_SPECS knows about, even ones with no data.

    storage/db.py::conn() only creates the tables in its own SCHEMA string --
    paper_observations/paper_outcomes/paper_failure_analysis are each created
    by their own module's ensure_*_table(). A restore must produce a database
    that looks like a fully-migrated one regardless of which of those tables
    happen to be empty, so it calls all of them before touching data.
    """
    from intraday_engine.research.monitoring import ensure_monitoring_table
    from intraday_engine.research.paper_diagnostics import ensure_diagnostics_table
    from intraday_engine.research.paper_learning import ensure_learning_table
    from intraday_engine.research.paper_observer import ensure_observation_table
    from intraday_engine.research.paper_outcomes import ensure_outcome_table

    open_connection(path=target_db_path).close()  # applies storage/db.py::SCHEMA
    ensure_observation_table(target_db_path)
    ensure_outcome_table(target_db_path)
    ensure_learning_table(target_db_path)
    ensure_diagnostics_table(target_db_path)
    ensure_monitoring_table(target_db_path)


class WarehouseRestoreError(RuntimeError):
    """Raised when the warehouse is missing, incompatible, or its contents don't verify."""


def _verify_manifest(root: Path, manifest: dict[str, dict]) -> dict[str, list[tuple[str, dict]]]:
    """Check every manifest entry's file exists and checksums correctly.

    Returns entries grouped by table: {table: [(absolute_path, entry), ...]}.
    Raises on the first missing file or checksum mismatch -- nothing is loaded
    until every entry in the manifest has been proven intact.
    """
    by_table: dict[str, list[tuple[str, dict]]] = {}
    for rel_path, entry in manifest.items():
        file_path = root / rel_path
        if not file_path.exists():
            raise WarehouseRestoreError(f"manifest references a missing partition file: {rel_path}")
        actual_checksum = sha256_of_file(file_path)
        if actual_checksum != entry["sha256"]:
            raise WarehouseRestoreError(
                f"checksum mismatch for {rel_path}: manifest says {entry['sha256']}, "
                f"file on disk hashes to {actual_checksum}"
            )
        by_table.setdefault(entry["table"], []).append((file_path.as_posix(), entry))
    return by_table


def restore_warehouse(root: str | Path, target_db_path: str) -> dict[str, int]:
    """Restore every table in `TABLE_SPECS` from the warehouse at `root` into a
    fresh DuckDB at `target_db_path`. Returns {table_name: rows_restored}.
    """
    root = Path(root)
    if not root.exists():
        raise WarehouseRestoreError(f"warehouse root does not exist: {root}")

    version_info = load_schema_version(root)
    if version_info is None:
        raise WarehouseRestoreError(
            f"{root} has no _schema_version.json -- refusing to trust an unversioned warehouse"
        )
    if version_info.get("schema_version") != SCHEMA_VERSION:
        raise WarehouseRestoreError(
            f"warehouse schema_version={version_info.get('schema_version')!r} does not match "
            f"code schema_version={SCHEMA_VERSION}"
        )

    manifest = load_manifest(root)
    verified_by_table = _verify_manifest(root, manifest)

    target_path = Path(target_db_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()

    _ensure_all_known_tables(target_db_path)

    restored_counts: dict[str, int] = {}
    connection = open_connection(path=target_db_path)
    try:
        for spec in TABLE_SPECS:
            entries = verified_by_table.get(spec.name, [])
            if not entries:
                restored_counts[spec.name] = 0
                continue

            expected_rows = sum(entry["rows"] for _, entry in entries)
            file_list_sql = ",".join(f"'{path}'" for path, _ in entries)
            # hive_partitioning=false: our partition directories are named
            # `date=YYYY-MM-DD` for human navigability, but every file already
            # carries the real partition column as an ordinary column -- letting
            # DuckDB auto-detect Hive partitioning would inject a *second*,
            # synthetic `date` column and break the positional column count
            # the target table expects.
            read_parquet_sql = f"read_parquet([{file_list_sql}], hive_partitioning=false)"
            actual_rows = connection.execute(f"SELECT COUNT(*) FROM {read_parquet_sql}").fetchone()[0]
            if actual_rows != expected_rows:
                raise WarehouseRestoreError(
                    f"{spec.name}: manifest reports {expected_rows} total rows across its "
                    f"{len(entries)} file(s), but read_parquet reports {actual_rows}"
                )

            connection.execute(f"INSERT INTO {spec.name} SELECT * FROM {read_parquet_sql}")
            restored_counts[spec.name] = actual_rows
    finally:
        connection.close()

    return restored_counts
