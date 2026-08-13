from __future__ import annotations

import json
from datetime import date

import pandas as pd

from intraday_engine.storage.db import conn


LEARNING_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_failure_analysis(
    observation_id VARCHAR PRIMARY KEY,
    evaluated_at TIMESTAMPTZ,
    trading_date DATE,
    symbol VARCHAR,
    side VARCHAR,
    outcome VARCHAR,
    pnl_points DOUBLE,
    r_multiple DOUBLE,
    signal_score DOUBLE,
    confidence DOUBLE,
    market_regime VARCHAR,
    market_score DOUBLE,
    scanner_rank INTEGER,
    candidate_score DOUBLE,
    relative_volume DOUBLE,
    price_change_pct DOUBLE,
    signal_reasons VARCHAR,
    signal_blockers VARCHAR,
    failure_class VARCHAR
);
"""


def ensure_learning_table(path: str | None = None) -> None:
    connection = conn(path)
    try:
        connection.execute(LEARNING_SCHEMA)
    finally:
        connection.close()


def _failure_class(outcome: str, side: str, reasons: str, blockers: str) -> str:
    if outcome in {"STOP", "STOP_GAP"}:
        try:
            blocker_values = json.loads(blockers or "[]")
        except json.JSONDecodeError:
            blocker_values = []
        try:
            reason_values = json.loads(reasons or "[]")
        except json.JSONDecodeError:
            reason_values = []
        # A genuine VWAP conflict is the trade firing *despite* VWAP
        # disagreeing with its direction (signals/engine.py always states one
        # of these two reasons, in whichever direction price actually sat --
        # matching the trade's own side is agreement, not conflict).
        vwap_conflict = (
            (side == "LONG" and "price is below VWAP" in reason_values)
            or (side == "SHORT" and "price is above VWAP" in reason_values)
        )
        if vwap_conflict:
            return "STOP_WITH_VWAP_CONFLICT"
        if any("volume" in str(x).lower() for x in blocker_values):
            return "STOP_WITH_WEAK_VOLUME"
        if any("extended" in str(x).lower() for x in blocker_values):
            return "STOP_WHILE_EXTENDED"
        return "STOP_OTHER"
    if outcome == "TIMEOUT":
        return "TIMEOUT"
    return "WIN"


def build_failure_analysis() -> int:
    """Create an auditable failure dataset without changing strategy parameters."""
    ensure_learning_table()
    connection = conn()
    try:
        rows = connection.execute(
            """
            SELECT o.observation_id, p.evaluated_at, o.trading_date, o.symbol,
                   p.side, p.outcome, p.pnl_points, p.r_multiple,
                   o.signal_score, o.confidence, o.market_regime, o.market_score,
                   o.scanner_rank, o.candidate_score, o.relative_volume,
                   o.price_change_pct, o.signal_reasons, o.signal_blockers
            FROM paper_outcomes p
            JOIN paper_observations o USING (observation_id)
            WHERE NOT EXISTS (
                SELECT 1 FROM paper_failure_analysis f
                WHERE f.observation_id = p.observation_id
            )
            ORDER BY p.evaluated_at
            """
        ).fetchall()
        if not rows:
            return 0
        columns = [
            "observation_id", "evaluated_at", "trading_date", "symbol", "side",
            "outcome", "pnl_points", "r_multiple", "signal_score", "confidence",
            "market_regime", "market_score", "scanner_rank", "candidate_score",
            "relative_volume", "price_change_pct", "signal_reasons", "signal_blockers",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        frame["failure_class"] = [
            _failure_class(o, s, r, b)
            for o, s, r, b in zip(
                frame["outcome"], frame["side"], frame["signal_reasons"], frame["signal_blockers"]
            )
        ]
        connection.register("paper_failure_analysis_in", frame)
        connection.execute("INSERT OR IGNORE INTO paper_failure_analysis SELECT * FROM paper_failure_analysis_in")
        return len(frame)
    finally:
        try:
            connection.unregister("paper_failure_analysis_in")
        except Exception:
            pass
        connection.close()


def learning_report(trading_date: date | None = None) -> dict:
    ensure_learning_table()
    connection = conn()
    try:
        date_filter = "" if trading_date is None else "WHERE trading_date = ?"
        params = [] if trading_date is None else [trading_date]
        # paper_failure_analysis denormalizes pnl_points/r_multiple from
        # paper_outcomes (both tables carry them), so unqualified references
        # here are ambiguous to the SQL binder -- qualify explicitly. mfe/mae
        # only exist on paper_outcomes, hence the join.
        summary = connection.execute(
            f"""
            SELECT failure_class, COUNT(*) AS trades,
                   AVG(f.r_multiple) AS avg_r,
                   SUM(f.pnl_points) AS net_points,
                   AVG(p.mfe_points) AS avg_mfe,
                   AVG(p.mae_points) AS avg_mae
            FROM paper_failure_analysis f
            JOIN paper_outcomes p USING (observation_id)
            {date_filter}
            GROUP BY failure_class
            ORDER BY trades DESC
            """,
            params,
        ).df()
        by_symbol = connection.execute(
            f"""
            SELECT symbol, side, COUNT(*) AS trades,
                   AVG(r_multiple) AS avg_r,
                   SUM(pnl_points) AS net_points
            FROM paper_failure_analysis
            {date_filter}
            GROUP BY symbol, side
            ORDER BY net_points DESC
            """,
            params,
        ).df()
        return {
            "failure_classes": summary.to_dict(orient="records"),
            "by_symbol_side": by_symbol.to_dict(orient="records"),
        }
    finally:
        connection.close()
