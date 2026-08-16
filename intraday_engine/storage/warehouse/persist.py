"""Export a DuckDB research database into the partitioned Parquet warehouse.

Every write goes: query source -> COPY to a `.tmp` file -> re-read the temp
file's row count and compare to the source query's row count -> sha256 the
file -> only then compare that checksum to the manifest's previously
recorded one to decide skip-vs-write, atomically rename it into place, and
record it in the manifest. A partition is never recorded (and never left in
place under its final name) unless it has already been proven to match the
source.

The skip decision is a checksum comparison, not just a row-count comparison:
a row revised in place (e.g. Upstox correcting a candle's OHLCV after
ingestion already stored it) changes the partition's content without
changing its row count, and a row-count-only check would silently keep
serving the stale, already-persisted version of that partition forever. The
tmp file is always (re-)written and hashed; it's only kept if the hash
actually differs from what's on record -- this is what makes repeated
persist calls idempotent (no-op when truly nothing changed) while still
catching in-place revisions (write when the content changed even if the
count didn't).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from intraday_engine.storage.warehouse.manifest import (
    load_manifest,
    load_schema_version,
    save_manifest,
    save_schema_version,
    sha256_of_file,
)
from intraday_engine.storage.warehouse.schema import MIN_COMPATIBLE_SCHEMA_VERSION, SCHEMA_VERSION, TABLE_SPECS


class WarehouseSchemaVersionError(RuntimeError):
    """Raised when the on-disk warehouse's schema version doesn't match the code's."""


class WarehousePersistError(RuntimeError):
    """Raised when a written partition fails verification against its source query."""


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
    )


def _distinct_partitions(connection: duckdb.DuckDBPyConnection, spec) -> list[tuple[object, int]]:
    """Return [(partition_value, source_row_count), ...] currently present in `spec`'s table.

    Some tables (paper_observations/outcomes/failure_analysis) are only ever
    created by their own module's ensure_*_table() the first time it runs, so
    a source database that hasn't exercised that path yet may not have them --
    treat a missing table the same as an existing-but-empty one.
    """
    if not _table_exists(connection, spec.name):
        return []
    if spec.partition_expr is None:
        count = connection.execute(f"SELECT COUNT(*) FROM {spec.name}").fetchone()[0]
        return [(None, count)] if count else []
    rows = connection.execute(
        f"SELECT {spec.partition_expr} AS partition_value, COUNT(*) "
        f"FROM {spec.name} GROUP BY partition_value ORDER BY partition_value"
    ).fetchall()
    return [(value, count) for value, count in rows]


def _partition_label(value: object) -> str:
    if value is None:
        return "dimension"
    return f"date={value.isoformat()}" if hasattr(value, "isoformat") else f"date={value}"


def _relative_path(spec, label: str) -> str:
    return f"{spec.zone}/{spec.name}/{label}/part.parquet"


def _select_sql(spec, value: object) -> str:
    if spec.partition_expr is None:
        return f"SELECT * FROM {spec.name}"
    literal = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return f"SELECT * FROM {spec.name} WHERE {spec.partition_expr} = DATE '{literal}'"


def persist_warehouse(source_db_path: str, root: str | Path) -> dict:
    """Export every table known to `TABLE_SPECS` from `source_db_path` into `root`.

    Opens the source read-only -- persisting never mutates the research database.
    Returns a summary dict: {"written": [...], "skipped": [...], "tables": {name: {...}}}.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    existing_version = load_schema_version(root)
    if existing_version is not None:
        on_disk_version = existing_version.get("schema_version")
        if not isinstance(on_disk_version, int) or not (MIN_COMPATIBLE_SCHEMA_VERSION <= on_disk_version <= SCHEMA_VERSION):
            raise WarehouseSchemaVersionError(
                f"warehouse at {root} was written with schema_version={on_disk_version!r}, which "
                f"is outside the range the running code (schema_version={SCHEMA_VERSION}) can "
                f"safely persist into ([{MIN_COMPATIBLE_SCHEMA_VERSION}, {SCHEMA_VERSION}])."
            )

    manifest = load_manifest(root)
    summary: dict = {"written": [], "skipped": [], "tables": {}}

    connection = duckdb.connect(source_db_path, read_only=True)
    try:
        for spec in TABLE_SPECS:
            table_summary = {"written": 0, "skipped": 0}
            for value, source_rows in _distinct_partitions(connection, spec):
                label = _partition_label(value)
                rel_path = _relative_path(spec, label)
                existing_entry = manifest.get(rel_path)

                target_path = root / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = target_path.with_name(target_path.name + ".tmp")

                connection.execute(
                    f"COPY ({_select_sql(spec, value)}) TO '{tmp_path.as_posix()}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                written_rows = connection.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{tmp_path.as_posix()}', hive_partitioning=false)"
                ).fetchone()[0]
                if written_rows != source_rows:
                    tmp_path.unlink(missing_ok=True)
                    raise WarehousePersistError(
                        f"{rel_path}: source query produced {source_rows} rows but the "
                        f"written parquet file contains {written_rows} -- refusing to record it"
                    )

                checksum = sha256_of_file(tmp_path)

                if existing_entry is not None and existing_entry.get("sha256") == checksum:
                    # Row count may or may not match -- what matters is the
                    # content is byte-for-byte identical to what's already
                    # durable, so there is genuinely nothing new to persist.
                    tmp_path.unlink(missing_ok=True)
                    table_summary["skipped"] += 1
                    summary["skipped"].append(rel_path)
                    continue

                tmp_path.replace(target_path)  # atomic within the same filesystem

                manifest[rel_path] = {
                    "table": spec.name,
                    "zone": spec.zone,
                    "partition": label,
                    "rows": written_rows,
                    "sha256": checksum,
                }
                # Save immediately: if the process is killed on the very next
                # partition, this one must still be recorded as done, not
                # silently forgotten (and redundantly-but-not-incorrectly
                # redone) on the next retry.
                save_manifest(root, manifest)
                table_summary["written"] += 1
                summary["written"].append(rel_path)
            summary["tables"][spec.name] = table_summary
    finally:
        connection.close()

    # The schema version is written only after every table/partition has been
    # processed successfully. This is the actual v4 -> v5 upgrade point: a
    # compatible v4 warehouse remains stamped v4 if a persist crashes partway
    # through, so a retry can still recognize it as the old-but-compatible
    # artifact. A completed persist stamps the warehouse v5, including the
    # newly-known instrument_master_history table (which is empty on a restore
    # of a genuine v4 warehouse until new mappings are observed).
    save_schema_version(root, SCHEMA_VERSION)

    return summary
