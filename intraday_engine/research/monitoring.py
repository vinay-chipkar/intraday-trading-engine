"""Automated, idempotent operational health monitoring for the unattended
research pipeline (ingestion -> paper trading -> learning data -> diagnostics
-> warehouse persist).

This is deliberately separate from research/paper_diagnostics.py:
paper_diagnostics answers "what does the accumulated trading sample say"
(a strategy/statistics question), while this module answers "is the pipeline
itself still working" (an operations question) -- ingestion failure trends,
stale-tick rates, data-quality flags, and how fast the evaluated sample is
actually growing toward a size that would support ML training.

It never touches signal/strategy/threshold logic and never trains a model.
Every call either produces a brand-new, timestamped `run_id` row (historical
reports are permanent, never overwritten) or, if nothing has changed since
the last report, is a genuine no-op. When the accumulated history shows a
sustained failure pattern (ingestion has failed on every recent run), it
raises PipelineUnhealthy so the caller (a GitHub Actions step) can fail the
workflow loudly instead of quietly continuing to run on broken data.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from intraday_engine.research.paper_diagnostics import MIN_SAMPLE_SIZE
from intraday_engine.storage.db import conn

MONITORING_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_monitoring(
    run_id VARCHAR PRIMARY KEY,
    generated_at TIMESTAMPTZ,
    healthy BOOLEAN,
    report_json VARCHAR
);
"""


class PipelineUnhealthy(RuntimeError):
    """Raised when accumulated operational history shows a sustained failure."""


def ensure_monitoring_table(path: str | None = None) -> None:
    connection = conn(path)
    try:
        connection.execute(MONITORING_SCHEMA)
    finally:
        connection.close()


def _ingestion_health(lookback_days: int) -> dict:
    connection = conn()
    try:
        rows = connection.execute(
            """
            SELECT run_id, started_at, status, error
            FROM ingestion_runs
            WHERE started_at >= now() - INTERVAL '{days} days'
            ORDER BY started_at
            """.format(days=int(lookback_days))
        ).fetchall()
    finally:
        connection.close()

    total_runs = len(rows)
    failed_runs = [r for r in rows if r[2] != "SUCCESS"]
    consecutive_failures = 0
    for row in reversed(rows):
        if row[2] == "SUCCESS":
            break
        consecutive_failures += 1

    return {
        "lookback_days": lookback_days,
        "total_runs": total_runs,
        "failed_runs": len(failed_runs),
        "failure_rate": (len(failed_runs) / total_runs) if total_runs else None,
        "consecutive_failures": consecutive_failures,
        "latest_run_id": rows[-1][0] if rows else None,
        "latest_errors": [r[3] for r in failed_runs[-3:] if r[3]],
    }


def _data_quality_summary(lookback_days: int) -> dict:
    connection = conn()
    try:
        rows = connection.execute(
            """
            SELECT issue_type, COUNT(*)
            FROM data_quality_events
            WHERE event_time >= now() - INTERVAL '{days} days'
            GROUP BY issue_type
            """.format(days=int(lookback_days))
        ).fetchall()
    finally:
        connection.close()
    return {"lookback_days": lookback_days, "events_by_type": {issue: count for issue, count in rows}}


def _stale_tick_rate(lookback_days: int) -> dict:
    connection = conn()
    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'STALE_DATA'),
                COUNT(*)
            FROM paper_observations
            WHERE observed_at >= now() - INTERVAL '{days} days'
            """.format(days=int(lookback_days))
        ).fetchone()
    finally:
        connection.close()
    stale, total = row
    return {
        "lookback_days": lookback_days,
        "stale_observations": stale,
        "total_observations": total,
        "stale_rate": (stale / total) if total else None,
    }


def _sample_growth(lookback_days: int) -> dict:
    connection = conn()
    try:
        rows = connection.execute(
            """
            SELECT generated_at, sample_count
            FROM research_diagnostics
            WHERE generated_at >= now() - INTERVAL '{days} days'
            ORDER BY generated_at
            """.format(days=int(lookback_days))
        ).fetchall()
        latest = connection.execute(
            "SELECT sample_count FROM research_diagnostics ORDER BY generated_at DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()

    current_sample_count = latest[0] if latest else 0
    if len(rows) < 2:
        return {
            "lookback_days": lookback_days,
            "current_sample_count": current_sample_count,
            "growth_per_day": None,
            "projected_days_to_min_sample": None,
            "min_sample_size": MIN_SAMPLE_SIZE,
        }

    first_at, first_count = rows[0]
    last_at, last_count = rows[-1]
    elapsed_days = (last_at - first_at).total_seconds() / 86400
    delta = last_count - first_count
    growth_per_day = (delta / elapsed_days) if elapsed_days > 0 else None

    projected_days = None
    if growth_per_day and growth_per_day > 0 and current_sample_count < MIN_SAMPLE_SIZE:
        projected_days = (MIN_SAMPLE_SIZE - current_sample_count) / growth_per_day

    return {
        "lookback_days": lookback_days,
        "current_sample_count": current_sample_count,
        "growth_per_day": growth_per_day,
        "projected_days_to_min_sample": projected_days,
        "min_sample_size": MIN_SAMPLE_SIZE,
    }


def _latest_report_fingerprint() -> tuple | None:
    connection = conn()
    try:
        row = connection.execute(
            "SELECT report_json FROM research_monitoring ORDER BY generated_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        parsed = json.loads(row[0])
        return (parsed["ingestion"]["latest_run_id"], parsed["sample_growth"]["current_sample_count"])
    finally:
        connection.close()


def _persist_report(run_id: str, generated_at: datetime, healthy: bool, report: dict) -> None:
    connection = conn()
    try:
        connection.execute(
            "INSERT INTO research_monitoring VALUES (?, ?, ?, ?)",
            [run_id, generated_at, healthy, json.dumps(report, indent=2, default=str)],
        )
    finally:
        connection.close()


def build_monitoring_report(
    *,
    lookback_days: int = 7,
    sample_growth_lookback_days: int = 90,
    consecutive_failure_threshold: int = 3,
    force: bool = False,
    fail_on_unhealthy: bool = True,
) -> dict:
    """Build (and persist, as a new row) an operational health report.

    Idempotent: if nothing has changed since the last report (same latest
    ingestion run and same current sample count), this is a no-op and returns
    {"skipped": True, ...} without writing a new row. Pass force=True to
    bypass that check.

    Raises PipelineUnhealthy (after persisting the report) when the
    accumulated history shows a sustained failure -- ingestion failing on
    `consecutive_failure_threshold` or more runs in a row -- unless
    fail_on_unhealthy=False.
    """
    ensure_monitoring_table()

    ingestion = _ingestion_health(lookback_days)
    data_quality = _data_quality_summary(lookback_days)
    stale_ticks = _stale_tick_rate(lookback_days)
    sample_growth = _sample_growth(sample_growth_lookback_days)

    fingerprint = (ingestion["latest_run_id"], sample_growth["current_sample_count"])
    if not force:
        previous = _latest_report_fingerprint()
        if previous is not None and previous == fingerprint:
            return {
                "skipped": True,
                "reason": "no new ingestion runs or sample growth since the last report",
            }

    reasons = []
    if ingestion["consecutive_failures"] >= consecutive_failure_threshold:
        reasons.append(
            f"ingestion has failed on the last {ingestion['consecutive_failures']} consecutive "
            f"run(s) (threshold={consecutive_failure_threshold})"
        )
    healthy = not reasons

    run_id = f"mon-{uuid.uuid4().hex}"
    generated_at = datetime.now(timezone.utc)
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "healthy": healthy,
        "reasons": reasons,
        "ingestion": ingestion,
        "data_quality": data_quality,
        "stale_ticks": stale_ticks,
        "sample_growth": sample_growth,
    }
    _persist_report(run_id, generated_at, healthy, report)
    report["skipped"] = False

    if not healthy and fail_on_unhealthy:
        raise PipelineUnhealthy("; ".join(reasons))
    return report
