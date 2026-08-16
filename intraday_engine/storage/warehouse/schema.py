"""Table-to-warehouse mapping and schema version for the Parquet warehouse.

This is the single source of truth for how each DuckDB table maps onto the
partitioned Parquet layout on disk (zone + partition key).

restore.py reads a table's partitions with union_by_name=true and inserts
BY NAME, so a table gaining a new *nullable* column (old partitions simply
restore with NULL for it, matching what those rows actually had) does NOT
require a SCHEMA_VERSION bump -- this was verified against real production
partitions (paper_observations/paper_outcomes already had data persisted
before they gained provenance columns like strategy_version).

Bump SCHEMA_VERSION for anything BY NAME can't paper over: a column being
removed, renamed, or changing type, or a table being restructured -- cases
where "restore what's there under its old name" would be actively wrong
rather than just incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass

# v2: feature_snapshots and training_labels gained an observation_id column
# (research/learning_pipeline.py) linking each row back to the paper
# observation it was derived from.
# v3: new research_diagnostics table (research/paper_diagnostics.py), an
# append-only ledger of aggregate diagnostics reports over evaluated paper
# observations. No warehouse has been bootstrapped yet (Phase 4's
# data-warehouse artifact chain hasn't had its first real run), so these are
# free bumps -- nothing existing needs migrating.
# v4: new research_monitoring table (research/monitoring.py), an append-only
# ledger of pipeline operational health reports (ingestion failure trends,
# stale-tick rate, sample growth). Still no real warehouse in production yet.
# v5: new instrument_master_history table (storage/db.py::upsert_instruments),
# an append-only log of every symbol -> instrument_key pair ever resolved --
# the audit trail for tracing a historical observation back to the exact
# instrument mapping in effect when it was recorded. A genuinely new table,
# not a column addition, so existing instrument_master partitions are
# unaffected and need no migration.
SCHEMA_VERSION = 5

# The oldest on-disk schema_version that restore.py/persist.py will still
# accept, rather than refuse outright. Every bump from this version up to
# SCHEMA_VERSION (v4 -> v5 above) is purely additive -- a brand new,
# previously-nonexistent table -- which restore.py already handles for any
# table absent from an older manifest (nothing to restore, so the table is
# simply left empty) and persist.py handles by treating a missing source
# table as having zero partitions. Nothing in that range removed, renamed, or
# restructured an existing table, so there is no real migration to perform:
# an old warehouse is safe to both restore from and persist into, and gets
# stamped up to SCHEMA_VERSION the next time persist_warehouse() writes to
# it. This floor exists because the scheduled paper-research workflow keeps
# a real v4 warehouse artifact around in production (see
# schema_version_compatibility tests) -- it must survive this bump rather
# than being rejected on the next scheduled run. Move this floor forward
# only when a future bump genuinely breaks backward compatibility (a column
# removal/rename/restructure), never just because time has passed.
MIN_COMPATIBLE_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class TableSpec:
    name: str
    zone: str  # "raw" | "research" | "ml"
    # SQL expression (evaluated against the source table) that produces the
    # DATE to partition by, or None for a small dimension table that is kept
    # as a single, always-overwritten file instead of date partitions.
    partition_expr: str | None


TABLE_SPECS: tuple[TableSpec, ...] = (
    # raw market data
    TableSpec("candles", "raw", "CAST(timestamp AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
    TableSpec("market_context", "raw", "trading_date"),
    TableSpec("market_news", "raw", "CAST(captured_at AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
    TableSpec("instrument_master", "raw", None),
    TableSpec("instrument_master_history", "raw", None),
    # research/decision data
    TableSpec("candidate_events", "research", "trading_date"),
    TableSpec("signals", "research", "trading_date"),
    TableSpec("research_runs", "research", "trading_date"),
    TableSpec("data_quality_events", "research", "CAST(event_time AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
    TableSpec("ingestion_runs", "research", "CAST(started_at AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
    TableSpec("paper_observations", "research", "trading_date"),
    TableSpec("paper_outcomes", "research", "CAST(evaluated_at AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
    TableSpec("paper_failure_analysis", "research", "trading_date"),
    TableSpec("research_diagnostics", "research", "CAST(generated_at AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
    TableSpec("research_monitoring", "research", "CAST(generated_at AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
    # ML training data
    TableSpec("feature_snapshots", "ml", "trading_date"),
    TableSpec("training_labels", "ml", "CAST(event_time AT TIME ZONE 'Asia/Kolkata' AS DATE)"),
)

TABLE_SPECS_BY_NAME: dict[str, TableSpec] = {spec.name: spec for spec in TABLE_SPECS}
