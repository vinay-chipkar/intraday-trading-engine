from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.research.paper_diagnostics import (
    MIN_SAMPLE_SIZE,
    build_diagnostics_report,
    ensure_diagnostics_table,
)
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
    return fake_settings.duckdb_path


def _insert_evaluated(
    *,
    observation_id: str,
    action: str = "BUY",
    candidate_score: float = 55.0,
    signal_score: float = 65.0,
    relative_volume: float = 1.2,
    market_regime: str = "NEUTRAL",
    market_context_status: str = "AVAILABLE",
    outcome: str = "TARGET",
    pnl_points: float = 1.0,
    r_multiple: float = 1.0,
    evaluated_at: str = "2026-08-11 06:00:00+00",
    feature: dict | None = None,
) -> None:
    connection = db.conn()
    try:
        connection.execute(
            """
            INSERT INTO paper_observations
                (observation_id, observed_at, bar_time, trading_date, symbol, instrument_key,
                 scanner_rank, candidate_score, price_change_pct, relative_volume, vwap,
                 market_regime, market_score, signal_action, signal_score, confidence,
                 entry_price, stop_loss, target, signal_reasons, signal_blockers, status,
                 market_context_status)
            VALUES
                (?, now(), TIMESTAMPTZ '2026-08-11 05:00:00+00', DATE '2026-08-11', 'TEST', 'NSE_EQ|TEST',
                 1, ?, 1.0, ?, 100.0,
                 ?, 0.0, ?, ?, ?,
                 100.0, 99.0, 101.0, '[]', '[]', 'EVALUATED', ?)
            """,
            [
                observation_id, candidate_score, relative_volume, market_regime, action, signal_score, signal_score,
                market_context_status,
            ],
        )
        connection.execute(
            f"""
            INSERT INTO paper_outcomes VALUES
                (?, TIMESTAMPTZ '{evaluated_at}', TIMESTAMPTZ '2026-08-11 05:01:00+00',
                 TIMESTAMPTZ '{evaluated_at}', 'LONG', 100.0, {100.0 + pnl_points}, 99.0, 101.0,
                 ?, ?, ?, 5, 0.5, -0.1, '1.0.0')
            """,
            [observation_id, outcome, pnl_points, r_multiple],
        )
        if feature is not None:
            connection.execute(
                """
                INSERT INTO feature_snapshots
                    (event_time, trading_date, symbol, instrument_key, timeframe, close, volume,
                     rsi14, vwap, ema9, ema20, ema50, trend, relative_volume, feature_score, observation_id)
                VALUES
                    (TIMESTAMPTZ '2026-08-11 05:00:00+00', DATE '2026-08-11', 'TEST', 'NSE_EQ|TEST', '1m',
                     ?, 1000, ?, ?, ?, ?, ?, ?, ?, 50.0, ?)
                """,
                [
                    feature.get("close", 100.0), feature.get("rsi14", 55.0), feature.get("vwap", 100.0),
                    feature.get("ema9", 100.0), feature.get("ema20", 99.0), feature.get("ema50", 98.0),
                    feature.get("trend", "UPTREND"), feature.get("relative_volume"), observation_id,
                ],
            )
    finally:
        connection.close()


def test_empty_population_produces_a_well_formed_zero_sample_report(isolated_db):
    report = build_diagnostics_report()
    assert report["sample_count"] == 0
    assert report["overall"]["sufficient_sample"] is False
    assert report["overall"]["trades"] == 0
    assert report["breakdowns"]["action"]["summary"] == []
    assert report["breakdowns"]["action"]["significant_vs_rest"] == []


def test_single_observation_is_flagged_as_insufficient_sample(isolated_db):
    _insert_evaluated(observation_id="obs-1")
    report = build_diagnostics_report()
    assert report["sample_count"] == 1
    assert report["overall"]["sufficient_sample"] is False
    for breakdown in report["breakdowns"].values():
        for row in breakdown["summary"]:
            assert row["sufficient_sample"] is False
        assert breakdown["significant_vs_rest"] == []  # never claims significance below min_n


def test_repeated_run_with_no_new_data_is_skipped_not_duplicated(isolated_db):
    _insert_evaluated(observation_id="obs-1")
    first = build_diagnostics_report()
    second = build_diagnostics_report()

    assert first.get("skipped", False) is False
    assert second["skipped"] is True
    assert second["reason"] == "no new evaluated observations since the last report"

    connection = db.conn()
    count = connection.execute("SELECT COUNT(*) FROM research_diagnostics").fetchone()[0]
    connection.close()
    assert count == 1


def test_new_evaluated_data_produces_a_new_row_never_overwriting_the_old_one(isolated_db):
    _insert_evaluated(observation_id="obs-1")
    first = build_diagnostics_report()

    _insert_evaluated(observation_id="obs-2", evaluated_at="2026-08-12 06:00:00+00")
    second = build_diagnostics_report()

    assert second["skipped"] is False
    assert second["run_id"] != first["run_id"]
    assert second["sample_count"] == 2

    connection = db.conn()
    rows = connection.execute(
        "SELECT run_id, sample_count FROM research_diagnostics ORDER BY generated_at"
    ).fetchall()
    connection.close()
    assert len(rows) == 2  # both historical reports are preserved
    assert rows[0] == (first["run_id"], 1)
    assert rows[1] == (second["run_id"], 2)


def test_force_bypasses_the_idempotency_skip(isolated_db):
    _insert_evaluated(observation_id="obs-1")
    build_diagnostics_report()
    forced = build_diagnostics_report(force=True)

    assert forced["skipped"] is False
    connection = db.conn()
    count = connection.execute("SELECT COUNT(*) FROM research_diagnostics").fetchone()[0]
    connection.close()
    assert count == 2  # a second, identical-content report row was still added


def test_breakdown_by_signal_direction_matches_hand_computed_stats(isolated_db):
    _insert_evaluated(observation_id="buy-1", action="BUY", outcome="TARGET", pnl_points=1.0, r_multiple=1.0)
    _insert_evaluated(observation_id="buy-2", action="BUY", outcome="STOP", pnl_points=-1.0, r_multiple=-1.0)
    _insert_evaluated(observation_id="sell-1", action="SELL", outcome="TARGET", pnl_points=2.0, r_multiple=2.0)

    report = build_diagnostics_report(min_sample_size=1)
    by_action = {row["action"]: row for row in report["breakdowns"]["action"]["summary"]}

    assert by_action["BUY"]["trades"] == 2
    assert by_action["BUY"]["wins"] == 1
    assert by_action["BUY"]["expectancy_r"] == pytest.approx(0.0)
    assert by_action["SELL"]["trades"] == 1
    assert by_action["SELL"]["win_rate_pct"] == pytest.approx(100.0)


def test_significance_is_only_reported_once_min_sample_size_is_met():
    # Build a population where "BUY" is a clear, consistent loser and "SELL" a
    # clear, consistent winner, with min_sample_size lowered so this test
    # doesn't need 30+ rows to exercise the significance path.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        import intraday_engine.storage.db as db_module
        fake_settings = dataclasses.replace(real_settings, duckdb_path=f"{tmp}/t.duckdb")
        db_module.settings = fake_settings
        ensure_observation_table()
        ensure_outcome_table()
        ensure_learning_table()
        ensure_diagnostics_table()

        for i in range(10):
            _insert_evaluated(
                observation_id=f"buy-{i}", action="BUY", outcome="STOP",
                pnl_points=-1.0 + i * 0.01, r_multiple=-1.0 + i * 0.01,
            )
        for i in range(10):
            _insert_evaluated(
                observation_id=f"sell-{i}", action="SELL", outcome="TARGET",
                pnl_points=1.5 - i * 0.01, r_multiple=1.5 - i * 0.01,
            )

        report = build_diagnostics_report(min_sample_size=10)
        significance = {row["action"]: row for row in report["breakdowns"]["action"]["significant_vs_rest"]}

        assert "BUY" in significance and "SELL" in significance
        assert significance["BUY"]["p_value"] < 0.01
        assert significance["BUY"]["diff"] < 0
        assert significance["SELL"]["diff"] > 0


def test_vwap_and_ema_conditions_derived_from_feature_snapshot(isolated_db):
    _insert_evaluated(
        observation_id="obs-1",
        feature={"close": 106.0, "vwap": 100.0, "ema9": 105.0, "ema20": 103.0, "ema50": 101.0},
    )
    report = build_diagnostics_report()
    vwap_rows = {row["vwap_condition"]: row for row in report["breakdowns"]["vwap_condition"]["summary"]}
    ema_rows = {row["ema_alignment"]: row for row in report["breakdowns"]["ema_alignment"]["summary"]}
    assert "ABOVE_VWAP" in vwap_rows
    assert "BULLISH_ALIGNED" in ema_rows


def test_market_context_missing_is_reported_separately_from_genuine_neutral(isolated_db):
    _insert_evaluated(
        observation_id="obs-available", market_regime="NEUTRAL", market_context_status="AVAILABLE",
        outcome="TARGET", pnl_points=1.0, r_multiple=1.0,
    )
    _insert_evaluated(
        observation_id="obs-missing", market_regime="NEUTRAL", market_context_status="MARKET_CONTEXT_MISSING",
        outcome="STOP", pnl_points=-1.0, r_multiple=-1.0,
    )
    report = build_diagnostics_report(min_sample_size=1)
    by_status = {row["market_context_status"]: row for row in report["breakdowns"]["market_context_status"]["summary"]}

    assert set(by_status) == {"AVAILABLE", "MARKET_CONTEXT_MISSING"}
    assert by_status["AVAILABLE"]["trades"] == 1
    assert by_status["MARKET_CONTEXT_MISSING"]["trades"] == 1
    # Same market_regime ("NEUTRAL") for both -- only the explicit status
    # dimension separates a genuinely neutral market from a missing capture.
    assert by_status["AVAILABLE"]["win_rate_pct"] == pytest.approx(100.0)
    assert by_status["MARKET_CONTEXT_MISSING"]["win_rate_pct"] == pytest.approx(0.0)


def test_historical_rows_without_the_column_report_as_unknown_not_available(isolated_db):
    # A row predating this fix (market_context_status never populated) must
    # not be silently assumed AVAILABLE.
    _insert_evaluated(observation_id="obs-legacy")
    connection = db.conn()
    connection.execute(
        "UPDATE paper_observations SET market_context_status = NULL WHERE observation_id = 'obs-legacy'"
    )
    connection.close()

    report = build_diagnostics_report()
    statuses = {row["market_context_status"] for row in report["breakdowns"]["market_context_status"]["summary"]}
    assert statuses == {"unknown"}


def test_rvol_band_uses_the_live_feature_snapshot_not_the_frozen_scanner_snapshot(isolated_db):
    # Mirrors the real production case found in the Phase 8 review: the
    # scanner-time relative_volume stored on paper_observations was 0.0 for
    # every trade (no historical backfill yet), while the signal engine's own
    # live, bar-level RVOL -- stored on feature_snapshots -- was genuinely
    # elevated. rvol_band must reflect the value that actually drove the
    # signal, not the stale scanner snapshot.
    _insert_evaluated(
        observation_id="obs-1",
        relative_volume=0.0,
        feature={"relative_volume": 2.5},
    )
    report = build_diagnostics_report()
    rvol_rows = {row["rvol_band"] for row in report["breakdowns"]["rvol_band"]["summary"]}
    assert rvol_rows == {"2.0+"}


def test_observation_without_a_feature_snapshot_reports_unknown_not_a_crash(isolated_db):
    _insert_evaluated(observation_id="obs-1")  # no feature snapshot inserted
    report = build_diagnostics_report()
    vwap_rows = {row["vwap_condition"] for row in report["breakdowns"]["vwap_condition"]["summary"]}
    assert vwap_rows == {"unknown"}


def test_min_sample_size_constant_is_reasonable():
    assert MIN_SAMPLE_SIZE >= 20  # not so low that a handful of trades look "sufficient"
