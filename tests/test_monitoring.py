from __future__ import annotations

import dataclasses

import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.research.monitoring import (
    PipelineUnhealthy,
    build_monitoring_report,
    ensure_monitoring_table,
)
from intraday_engine.research.paper_diagnostics import ensure_diagnostics_table
from intraday_engine.research.paper_learning import ensure_learning_table
from intraday_engine.research.paper_observer import ensure_observation_table
from intraday_engine.research.paper_outcomes import ensure_outcome_table


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "test.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)
    ensure_observation_table()
    ensure_outcome_table()
    ensure_learning_table()
    ensure_diagnostics_table()
    ensure_monitoring_table()
    return fake_settings.duckdb_path


def _insert_ingestion_run(*, status: str, hours_ago: float, error: str | None = None) -> None:
    connection = db.conn()
    try:
        connection.execute(
            f"""
            INSERT INTO ingestion_runs VALUES
                ('ing-{hours_ago}-{status}', now() - INTERVAL '{hours_ago} hours',
                 now() - INTERVAL '{hours_ago} hours', 'PAPER', '1m', 10, 10, 100, 100, ?, ?)
            """,
            [status, error],
        )
    finally:
        connection.close()


def _insert_diagnostics_row(*, sample_count: int, hours_ago: float) -> None:
    connection = db.conn()
    try:
        connection.execute(
            f"""
            INSERT INTO research_diagnostics VALUES
                ('diag-{hours_ago}', now() - INTERVAL '{hours_ago} hours', ?, now(), '{{}}')
            """,
            [sample_count],
        )
    finally:
        connection.close()


def _insert_observation(*, status: str, hours_ago: float) -> None:
    connection = db.conn()
    try:
        connection.execute(
            f"""
            INSERT INTO paper_observations
                (observation_id, observed_at, bar_time, trading_date, symbol, instrument_key,
                 scanner_rank, candidate_score, price_change_pct, relative_volume, vwap,
                 market_regime, market_score, signal_action, signal_score, confidence,
                 entry_price, stop_loss, target, signal_reasons, signal_blockers, status)
            VALUES
                ('obs-{status}-{hours_ago}', now() - INTERVAL '{hours_ago} hours',
                 now() - INTERVAL '{hours_ago} hours', CURRENT_DATE, 'TEST', 'NSE_EQ|TEST',
                 1, 50.0, 1.0, 1.2, 100.0, 'NEUTRAL', 0.0, 'BUY', 60.0, 60.0,
                 100.0, 99.0, 101.0, '[]', '[]', ?)
            """,
            [status],
        )
    finally:
        connection.close()


def test_empty_history_is_healthy_with_no_growth_projection(isolated_db):
    report = build_monitoring_report()
    assert report["healthy"] is True
    assert report["ingestion"]["total_runs"] == 0
    assert report["sample_growth"]["growth_per_day"] is None
    assert report["sample_growth"]["projected_days_to_min_sample"] is None


def test_all_successful_runs_are_healthy(isolated_db):
    for h in (1, 2, 3):
        _insert_ingestion_run(status="SUCCESS", hours_ago=h)
    report = build_monitoring_report()
    assert report["healthy"] is True
    assert report["ingestion"]["consecutive_failures"] == 0
    assert report["ingestion"]["total_runs"] == 3


def test_sustained_failure_raises_pipeline_unhealthy(isolated_db):
    for h in (1, 2, 3):
        _insert_ingestion_run(status="FAILED", hours_ago=h, error="boom")
    with pytest.raises(PipelineUnhealthy, match="3 consecutive"):
        build_monitoring_report(consecutive_failure_threshold=3)

    connection = db.conn()
    healthy, = connection.execute("SELECT healthy FROM research_monitoring").fetchone()
    connection.close()
    assert healthy is False  # the report was still persisted before raising


def test_a_single_recent_failure_after_recovery_does_not_raise(isolated_db):
    _insert_ingestion_run(status="FAILED", hours_ago=5, error="transient")
    _insert_ingestion_run(status="SUCCESS", hours_ago=1)
    report = build_monitoring_report(consecutive_failure_threshold=3)
    assert report["healthy"] is True
    assert report["ingestion"]["consecutive_failures"] == 0


def test_fail_on_unhealthy_false_reports_without_raising(isolated_db):
    for h in (1, 2, 3):
        _insert_ingestion_run(status="FAILED", hours_ago=h, error="boom")
    report = build_monitoring_report(consecutive_failure_threshold=3, fail_on_unhealthy=False)
    assert report["healthy"] is False
    assert report["skipped"] is False


def test_repeated_call_with_no_new_data_is_skipped_not_duplicated(isolated_db):
    _insert_ingestion_run(status="SUCCESS", hours_ago=1)
    first = build_monitoring_report()
    second = build_monitoring_report()

    assert first["skipped"] is False
    assert second["skipped"] is True

    connection = db.conn()
    count = connection.execute("SELECT COUNT(*) FROM research_monitoring").fetchone()[0]
    connection.close()
    assert count == 1


def test_new_ingestion_run_produces_a_new_report_row(isolated_db):
    _insert_ingestion_run(status="SUCCESS", hours_ago=2)
    first = build_monitoring_report()
    _insert_ingestion_run(status="SUCCESS", hours_ago=1)
    second = build_monitoring_report()

    assert second["skipped"] is False
    assert second["run_id"] != first["run_id"]
    connection = db.conn()
    count = connection.execute("SELECT COUNT(*) FROM research_monitoring").fetchone()[0]
    connection.close()
    assert count == 2


def test_force_bypasses_the_idempotency_skip(isolated_db):
    _insert_ingestion_run(status="SUCCESS", hours_ago=1)
    build_monitoring_report()
    forced = build_monitoring_report(force=True)
    assert forced["skipped"] is False
    connection = db.conn()
    count = connection.execute("SELECT COUNT(*) FROM research_monitoring").fetchone()[0]
    connection.close()
    assert count == 2


def test_sample_growth_rate_and_projection_are_computed(isolated_db):
    _insert_diagnostics_row(sample_count=2, hours_ago=48)
    _insert_diagnostics_row(sample_count=10, hours_ago=0)
    report = build_monitoring_report()
    growth = report["sample_growth"]
    assert growth["current_sample_count"] == 10
    assert growth["growth_per_day"] == pytest.approx(4.0, rel=0.05)  # +8 samples over 2 days
    assert growth["projected_days_to_min_sample"] == pytest.approx(5.0, rel=0.05)  # (30-10)/4


def test_sample_growth_with_zero_or_negative_rate_has_no_projection(isolated_db):
    _insert_diagnostics_row(sample_count=10, hours_ago=48)
    _insert_diagnostics_row(sample_count=10, hours_ago=0)  # flat -- no growth
    report = build_monitoring_report()
    assert report["sample_growth"]["growth_per_day"] == pytest.approx(0.0, abs=1e-9)
    assert report["sample_growth"]["projected_days_to_min_sample"] is None


def test_stale_tick_rate_and_data_quality_events_are_reported(isolated_db):
    _insert_observation(status="STALE_DATA", hours_ago=1)
    _insert_observation(status="SIGNAL_PENDING", hours_ago=1)
    connection = db.conn()
    connection.execute(
        "INSERT INTO data_quality_events VALUES (now(), 'AAA', NULL, now(), 'INGESTION_QUALITY', 'x')"
    )
    connection.close()

    report = build_monitoring_report()
    assert report["stale_ticks"]["stale_observations"] == 1
    assert report["stale_ticks"]["total_observations"] == 2
    assert report["stale_ticks"]["stale_rate"] == pytest.approx(0.5)
    assert report["data_quality"]["events_by_type"]["INGESTION_QUALITY"] == 1


def test_lookback_window_excludes_old_ingestion_runs(isolated_db):
    _insert_ingestion_run(status="FAILED", hours_ago=24 * 30, error="ancient")  # 30 days old
    report = build_monitoring_report(lookback_days=7)
    assert report["ingestion"]["total_runs"] == 0
