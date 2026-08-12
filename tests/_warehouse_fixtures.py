"""Shared, not-collected-as-tests helper for warehouse persist/restore tests.

Builds a small source DuckDB using the app's real schema (via
intraday_engine.storage.db.conn) spanning two trading days and two zones
(raw + research), with a timezone-aware timestamp inserted using a non-UTC
offset specifically to exercise timestamp round-tripping.
"""

from __future__ import annotations

from intraday_engine.research.monitoring import ensure_monitoring_table
from intraday_engine.research.paper_diagnostics import ensure_diagnostics_table
from intraday_engine.research.paper_learning import ensure_learning_table
from intraday_engine.research.paper_observer import ensure_observation_table
from intraday_engine.research.paper_outcomes import ensure_outcome_table
from intraday_engine.storage.db import conn


def build_source_db(path: str) -> None:
    # A real production database also has these tables (each created by its
    # own module's ensure_*_table()), even before any paper session runs.
    ensure_observation_table(path)
    ensure_outcome_table(path)
    ensure_learning_table(path)
    ensure_diagnostics_table(path)
    ensure_monitoring_table(path)

    connection = conn(path=path)
    try:
        connection.execute(
            "INSERT INTO candles VALUES "
            "('NSE_EQ|AAA','AAA', TIMESTAMPTZ '2026-08-11 03:45:00+00','1m',100,101,99,100.5,1000,NULL),"
            "('NSE_EQ|AAA','AAA', TIMESTAMPTZ '2026-08-11 03:46:00+00','1m',100.5,102,99,101,1100,NULL),"
            "('NSE_EQ|AAA','AAA', TIMESTAMPTZ '2026-08-12 03:45:00+00','1m',101,103,100,102,1200,NULL),"
            "('NSE_EQ|BBB','BBB', TIMESTAMPTZ '2026-08-12 09:15:00+05:30','1m',50,51,49,50.5,500,NULL)"
        )
        connection.execute(
            "INSERT INTO instrument_master VALUES "
            "('AAA','NSE_EQ|AAA','AAA Ltd','AAA', TIMESTAMPTZ '2026-08-11 00:00:00+00'),"
            "('BBB','NSE_EQ|BBB','BBB Ltd','BBB', TIMESTAMPTZ '2026-08-11 00:00:00+00')"
        )
        connection.execute(
            "INSERT INTO candidate_events "
            "(event_time, trading_date, symbol, instrument_key, ltp, volume, relative_volume, "
            " price_change_pct, vwap, candidate_score, reason) VALUES "
            "(TIMESTAMPTZ '2026-08-11 05:00:00+00', DATE '2026-08-11', 'AAA', 'NSE_EQ|AAA', "
            " 100.5, 1000, 1.2, 0.5, 100.2, 55.0, 'MOMENTUM'),"
            "(TIMESTAMPTZ '2026-08-12 05:00:00+00', DATE '2026-08-12', 'BBB', 'NSE_EQ|BBB', "
            " 50.5, 500, 0.9, 1.0, 50.1, 40.0, 'NO_STRONG_FACTOR')"
        )
    finally:
        connection.close()
