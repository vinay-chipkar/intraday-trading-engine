"""Table-to-warehouse mapping and schema version for the Parquet warehouse.

This is the single source of truth for how each DuckDB table maps onto the
partitioned Parquet layout on disk (zone + partition key). Bump
SCHEMA_VERSION whenever a table gains/loses/renames a column in
`intraday_engine/storage/db.py`'s SCHEMA -- persist/restore both refuse to
mix data across schema versions rather than guessing at compatibility.
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
SCHEMA_VERSION = 4


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
