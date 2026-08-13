import dataclasses

import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.research import paper_learning
from intraday_engine.research.paper_learning import _failure_class
from intraday_engine.research.paper_outcomes import ensure_outcome_table


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point storage.db.conn() at a throwaway file so tests never touch data/trading.duckdb."""
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "test.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)


def test_vwap_agreement_is_not_a_conflict():
    # "price is above VWAP" on a LONG is VWAP *agreeing* with the trade --
    # merely mentioning VWAP must not be classified as a conflict. With no
    # real conflict present, this should fall through to the weak-volume
    # blocker instead.
    assert _failure_class(
        "STOP",
        "LONG",
        '["price is above VWAP"]',
        '["relative volume is weak"]',
    ) == "STOP_WITH_WEAK_VOLUME"


def test_long_stopped_out_despite_vwap_disagreeing_is_a_genuine_conflict():
    assert _failure_class(
        "STOP",
        "LONG",
        '["price is below VWAP"]',
        '[]',
    ) == "STOP_WITH_VWAP_CONFLICT"


def test_short_stopped_out_despite_vwap_disagreeing_is_a_genuine_conflict():
    assert _failure_class(
        "STOP",
        "SHORT",
        '["price is above VWAP"]',
        '[]',
    ) == "STOP_WITH_VWAP_CONFLICT"


def test_stop_with_extension_is_classified():
    assert _failure_class(
        "STOP_GAP",
        "LONG",
        '[]',
        '["price is extended from VWAP"]',
    ) == "STOP_WHILE_EXTENDED"


def test_timeout_and_win_are_classified():
    assert _failure_class("TIMEOUT", "LONG", "[]", "[]") == "TIMEOUT"
    assert _failure_class("TARGET", "LONG", "[]", "[]") == "WIN"


def test_learning_report_summary_does_not_raise_on_ambiguous_columns(isolated_db):
    # paper_failure_analysis denormalizes pnl_points/r_multiple from
    # paper_outcomes, so both tables carry them -- the summary query's
    # unqualified AVG(r_multiple)/SUM(pnl_points) raised "Ambiguous reference
    # to column name" from DuckDB as soon as any real outcome data existed.
    # No prior test exercised this SQL against a real DuckDB engine, so it was
    # only ever hit for the first time by real paper-trading outcomes.
    ensure_outcome_table()
    paper_learning.ensure_learning_table()
    connection = db.conn()
    try:
        connection.execute(
            """
            INSERT INTO paper_outcomes
                (observation_id, evaluated_at, entry_time, exit_time, side, entry_price,
                 exit_price, stop_loss, target, outcome, pnl_points, r_multiple,
                 holding_bars, mfe_points, mae_points)
            VALUES
                ('obs-1', now(), now(), now(), 'LONG', 100.0, 103.0, 98.0, 103.0,
                 'TARGET', 3.0, 1.5, 5, 3.0, -1.0)
            """
        )
        connection.execute(
            """
            INSERT INTO paper_failure_analysis
                (observation_id, evaluated_at, trading_date, symbol, side, outcome,
                 pnl_points, r_multiple, signal_score, confidence, market_regime,
                 market_score, scanner_rank, candidate_score, relative_volume,
                 price_change_pct, signal_reasons, signal_blockers, failure_class)
            VALUES
                ('obs-1', now(), current_date, 'TEST', 'LONG', 'TARGET',
                 3.0, 1.5, 70.0, 70.0, 'NEUTRAL',
                 0.0, 1, 50.0, 1.2,
                 0.5, '[]', '[]', 'WIN')
            """
        )
    finally:
        connection.close()

    report = paper_learning.learning_report()
    assert report["failure_classes"] == [
        {"failure_class": "WIN", "trades": 1, "avg_r": 1.5, "net_points": 3.0, "avg_mfe": 3.0, "avg_mae": -1.0}
    ]
    assert report["by_symbol_side"] == [
        {"symbol": "TEST", "side": "LONG", "trades": 1, "avg_r": 1.5, "net_points": 3.0}
    ]
