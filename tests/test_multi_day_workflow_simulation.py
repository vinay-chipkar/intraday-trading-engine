"""End-to-end simulation of the actual GitHub Actions workflow shape, run
across several consecutive "days":

    restore_warehouse (skipped on day 1 -- no warehouse yet)
    -> ingest candles -> paper observation -> evaluate_pending
    -> build_feature_snapshots_and_labels -> build_diagnostics_report
    -> build_monitoring_report -> persist_warehouse

Each simulated day uses a *fresh* DuckDB file, exactly like a fresh GitHub
Actions runner that has no local disk state except whatever
restore_warehouse pulls in from the (locally rooted, in this test) warehouse
artifact. This is the concrete verification the user asked for: that the
pipeline can run repeatedly, day after day, accumulating every table without
duplication, and that a brand-new runner can always recover the complete
research state from the warehouse alone.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.research.learning_pipeline import build_feature_snapshots_and_labels
from intraday_engine.research.monitoring import build_monitoring_report, ensure_monitoring_table
from intraday_engine.research.paper_diagnostics import build_diagnostics_report, ensure_diagnostics_table
from intraday_engine.research.paper_learning import ensure_learning_table
from intraday_engine.research.paper_observer import build_observation, ensure_observation_table, persist_observations
from intraday_engine.research.paper_outcomes import ensure_outcome_table, evaluate_pending
from intraday_engine.signals.engine import TradeSignal
from intraday_engine.storage.warehouse import persist_warehouse, restore_warehouse
from intraday_engine.storage.warehouse.schema import TABLE_SPECS


@pytest.fixture
def warehouse_root(tmp_path):
    return tmp_path / "warehouse"


def _day_db_path(tmp_path, day: int) -> str:
    # A distinct file per day -- nothing on disk carries over between days
    # except what restore_warehouse explicitly pulls back in, exactly like a
    # fresh GitHub Actions runner.
    return str(tmp_path / f"day{day}.duckdb")


def _rising_candles(symbol: str, n: int, *, trading_date: str, start_minute: int = 45, base: float = 100.0):
    rows = []
    price = base
    for i in range(n):
        minute = start_minute + i
        hour = 3 + minute // 60
        minute_of_hour = minute % 60
        ts = f"{trading_date} {hour:02d}:{minute_of_hour:02d}:00+00"
        price += 0.05
        rows.append((ts, price - 0.05, price + 0.2, price - 0.2, price, 1000))
    return rows


def _insert_candles(db_path: str, symbol: str, rows: list[tuple]) -> None:
    connection = db.conn(path=db_path)
    for ts, o, h, l, c, v in rows:
        connection.execute(
            f"INSERT INTO candles VALUES ('NSE_EQ|{symbol}','{symbol}', TIMESTAMPTZ '{ts}','1m',{o},{h},{l},{c},{v},NULL)"
        )
    connection.close()


def _ensure_all_tables(db_path: str) -> None:
    ensure_observation_table(db_path)
    ensure_outcome_table(db_path)
    ensure_learning_table(db_path)
    ensure_diagnostics_table(db_path)
    ensure_monitoring_table(db_path)


def _create_observation_and_outcome(db_path: str, symbol: str, bar_time: pd.Timestamp, entry: float) -> str:
    signal = TradeSignal(
        action="BUY", score=80.0, confidence=80.0, entry=entry,
        stop_loss=entry - 1.0, target=entry + 0.3,
        reward_risk=0.3, reasons=(), blockers=(), symbol=symbol, event_time=bar_time,
    )
    row = build_observation(
        observed_at=pd.Timestamp.now(tz="UTC"),
        trading_date=bar_time.date(),
        candidate={"symbol": symbol, "instrument_key": f"NSE_EQ|{symbol}", "rank": 1, "candidate_score": 50.0,
                   "change_pct": 1.0, "relative_volume": 1.5, "vwap": 100.0},
        signal=signal, market_regime="NEUTRAL", market_score=0.0, bar_time=bar_time,
    )
    persist_observations([row])
    return row["observation_id"]


def _run_one_day(db_path: str, *, day: int, warehouse_root, is_first_day: bool, monkeypatch) -> dict:
    """Runs exactly the workflow's afternoon-job data steps for one simulated
    day and returns the counters a caller can check for accumulation."""
    fake_settings = dataclasses.replace(real_settings, duckdb_path=db_path)
    monkeypatch.setattr(db, "settings", fake_settings)

    if is_first_day:
        _ensure_all_tables(db_path)  # first-ever run: nothing to restore
    else:
        restore_warehouse(str(warehouse_root), db_path)

    trading_date = f"2026-08-{10 + day:02d}"
    symbol = "TEST"
    rows = _rising_candles(symbol, 60, trading_date=trading_date)
    _insert_candles(db_path, symbol, rows)

    bars = db.conn(path=db_path).execute(
        "SELECT timestamp, close FROM candles WHERE symbol = ? ORDER BY timestamp", [symbol]
    ).df()
    # index of today's decision bar within *today's* 60 rows (bars accumulate
    # across days once restored, so anchor off the tail).
    todays_bars = bars.tail(60).reset_index(drop=True)
    bar_time = pd.Timestamp(todays_bars["timestamp"].iloc[30])
    entry = float(todays_bars["close"].iloc[30])

    _create_observation_and_outcome(db_path, symbol, bar_time, entry)
    evaluate_pending(max_holding_bars=30)
    learning_summary = build_feature_snapshots_and_labels()
    diagnostics_report = build_diagnostics_report()
    monitoring_report = build_monitoring_report(fail_on_unhealthy=False)

    persist_summary = persist_warehouse(db_path, str(warehouse_root))

    connection = db.conn(path=db_path)
    counts = {
        spec.name: connection.execute(f"SELECT COUNT(*) FROM {spec.name}").fetchone()[0]
        for spec in TABLE_SPECS
    }
    connection.close()

    return {
        "counts": counts,
        "learning_summary": learning_summary,
        "diagnostics_report": diagnostics_report,
        "monitoring_report": monitoring_report,
        "persist_summary": persist_summary,
    }


def test_three_day_cycle_accumulates_without_duplication_and_restores_fully(tmp_path, warehouse_root, monkeypatch):
    results = []
    for day in range(1, 4):
        db_path = _day_db_path(tmp_path, day)
        result = _run_one_day(
            db_path, day=day, warehouse_root=warehouse_root, is_first_day=(day == 1), monkeypatch=monkeypatch
        )
        results.append(result)

    # -- Day 1: exactly one observation, one outcome, one snapshot/label, one
    # diagnostics row, 60 candles.
    assert results[0]["counts"]["candles"] == 60
    assert results[0]["counts"]["paper_observations"] == 1
    assert results[0]["counts"]["paper_outcomes"] == 1
    assert results[0]["counts"]["feature_snapshots"] == 1
    assert results[0]["counts"]["training_labels"] == 1
    assert results[0]["counts"]["research_diagnostics"] == 1
    assert results[0]["learning_summary"]["feature_snapshots_written"] == 1

    # -- Day 2: candles ACCUMULATE (60 + 60), never duplicated; exactly one
    # new observation processed (not the day-1 one again).
    assert results[1]["counts"]["candles"] == 120
    assert results[1]["counts"]["paper_observations"] == 2
    assert results[1]["counts"]["paper_outcomes"] == 2
    assert results[1]["counts"]["feature_snapshots"] == 2
    assert results[1]["counts"]["training_labels"] == 2
    assert results[1]["learning_summary"]["feature_snapshots_written"] == 1  # only today's new one
    assert results[1]["learning_summary"]["evaluated_pending"] == 1
    # diagnostics history is APPENDED, never overwritten -- two permanent rows.
    assert results[1]["counts"]["research_diagnostics"] == 2
    assert not results[1]["diagnostics_report"]["skipped"]

    # -- Day 3: the same pattern continues for a third cycle, proving this
    # isn't a fluke of day1->day2 specifically.
    assert results[2]["counts"]["candles"] == 180
    assert results[2]["counts"]["paper_observations"] == 3
    assert results[2]["counts"]["paper_outcomes"] == 3
    assert results[2]["counts"]["feature_snapshots"] == 3
    assert results[2]["counts"]["training_labels"] == 3
    assert results[2]["counts"]["research_diagnostics"] == 3
    assert results[2]["learning_summary"]["feature_snapshots_written"] == 1

    # -- Every feature snapshot / training label maps to a distinct
    # observation -- no observation was ever double-processed across the
    # three restore/persist cycles.
    final_db = _day_db_path(tmp_path, 3)
    connection = db.conn(path=final_db)
    snapshot_obs = connection.execute("SELECT observation_id FROM feature_snapshots").df()
    label_obs = connection.execute("SELECT observation_id FROM training_labels").df()
    connection.close()
    assert snapshot_obs["observation_id"].nunique() == 3
    assert label_obs["observation_id"].nunique() == 3

    # -- A brand-new, never-before-seen runner restoring from the final
    # warehouse state recovers everything, byte-for-byte in row count, with
    # no duplication introduced by the restore itself.
    fresh_runner_db = str(tmp_path / "day4_fresh_runner.duckdb")
    fake_settings = dataclasses.replace(real_settings, duckdb_path=fresh_runner_db)
    monkeypatch.setattr(db, "settings", fake_settings)
    restored_counts = restore_warehouse(str(warehouse_root), fresh_runner_db)
    assert restored_counts["candles"] == 180
    assert restored_counts["paper_observations"] == 3
    assert restored_counts["paper_outcomes"] == 3
    assert restored_counts["feature_snapshots"] == 3
    assert restored_counts["training_labels"] == 3
    assert restored_counts["research_diagnostics"] == 3

    # -- Restoring a second time into yet another fresh runner produces
    # identical counts -- restore itself is not a source of drift/duplication.
    fresh_runner_db_2 = str(tmp_path / "day5_fresh_runner.duckdb")
    restored_counts_2 = restore_warehouse(str(warehouse_root), fresh_runner_db_2)
    assert restored_counts_2 == restored_counts


def test_monitoring_report_is_healthy_and_persisted_across_all_three_days(tmp_path, warehouse_root, monkeypatch):
    for day in range(1, 4):
        db_path = _day_db_path(tmp_path, day)
        result = _run_one_day(
            db_path, day=day, warehouse_root=warehouse_root, is_first_day=(day == 1), monkeypatch=monkeypatch
        )
        assert result["monitoring_report"]["healthy"] is True

    final_db = _day_db_path(tmp_path, 3)
    connection = db.conn(path=final_db)
    count = connection.execute("SELECT COUNT(*) FROM research_monitoring").fetchone()[0]
    connection.close()
    assert count == 3  # one permanent monitoring row per day, never overwritten
