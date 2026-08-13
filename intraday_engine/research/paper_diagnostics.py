"""Automated, idempotent research diagnostics over accumulated evaluated
paper-trading history.

This module only reads (paper_observations, paper_outcomes, feature_snapshots)
and writes one append-only ledger table (research_diagnostics). It never
touches signal/strategy/threshold logic, never trains a model, and never
mutates the tables it reads from.

Every breakdown is tagged `sufficient_sample` against MIN_SAMPLE_SIZE, and a
formal significance test (Welch's t-test vs. the rest of the population,
research/stats.py::cluster_significance) is only ever computed for levels
that clear that bar -- an underpowered comparison is not reported as if it
were a finding.

Each call either produces a brand-new, timestamped `run_id` row (never
overwrites or updates a previous report -- historical reports are permanent)
or, if nothing has changed since the last report (same sample count and same
latest evaluated_at), is a genuine no-op: safe to run on every workflow tick
with no manual bookkeeping and no report spam.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from intraday_engine.research.stats import (
    candidate_score_bucket,
    cluster_significance,
    rsi_bucket,
    rvol_bucket,
    score_bucket,
    summarize,
)
from intraday_engine.storage.db import conn

MIN_SAMPLE_SIZE = 30

DIAGNOSTICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_diagnostics(
    run_id VARCHAR PRIMARY KEY,
    generated_at TIMESTAMPTZ,
    sample_count INTEGER,
    evaluated_through TIMESTAMPTZ,
    report_json VARCHAR
);
"""

_BREAKDOWN_DIMENSIONS = (
    "action", "candidate_score_band", "signal_score_band", "rvol_band",
    "market_regime", "vwap_condition", "ema_alignment", "rsi_band", "trend",
)


def ensure_diagnostics_table(path: str | None = None) -> None:
    connection = conn(path)
    try:
        connection.execute(DIAGNOSTICS_SCHEMA)
    finally:
        connection.close()


def _load_evaluated_population() -> pd.DataFrame:
    connection = conn()
    try:
        return connection.execute(
            """
            SELECT
                o.observation_id, o.symbol, o.signal_action AS action, o.signal_score,
                o.candidate_score, o.market_regime,
                p.outcome, p.pnl_points, p.r_multiple, p.exit_time, p.evaluated_at,
                f.rsi14, f.vwap, f.close AS feature_close, f.ema9, f.ema20, f.ema50,
                f.trend, f.relative_volume
            FROM paper_observations o
            JOIN paper_outcomes p USING (observation_id)
            LEFT JOIN feature_snapshots f USING (observation_id)
            WHERE o.status = 'EVALUATED'
            """
        ).df()
    finally:
        connection.close()


def _vwap_condition(row: pd.Series) -> str:
    close, vwap = row.get("feature_close"), row.get("vwap")
    if pd.isna(close) or pd.isna(vwap) or vwap == 0:
        return "unknown"
    return "ABOVE_VWAP" if close > vwap else "BELOW_VWAP"


def _ema_alignment(row: pd.Series) -> str:
    close, e9, e20, e50 = row.get("feature_close"), row.get("ema9"), row.get("ema20"), row.get("ema50")
    if any(pd.isna(v) for v in (close, e9, e20, e50)):
        return "unknown"
    if close > e9 > e20 > e50 > 0:
        return "BULLISH_ALIGNED"
    if 0 < close < e9 < e20 < e50:
        return "BEARISH_ALIGNED"
    return "MIXED"


def _add_derived_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal_score_band"] = out["signal_score"].abs().apply(score_bucket)
    out["candidate_score_band"] = out["candidate_score"].apply(candidate_score_bucket)
    # feature_snapshots.relative_volume (the live, bar-level RVOL the signal
    # engine actually used), not paper_observations.relative_volume -- the
    # latter is a frozen snapshot from the scanner's once-per-day candidate
    # scan and is unrelated to the RVOL that drove any individual signal.
    out["rvol_band"] = out["relative_volume"].apply(rvol_bucket)
    out["rsi_band"] = out["rsi14"].apply(rsi_bucket)
    out["vwap_condition"] = out.apply(_vwap_condition, axis=1)
    out["ema_alignment"] = out.apply(_ema_alignment, axis=1)
    out["trend"] = out["trend"].fillna("unknown")
    out["market_regime"] = out["market_regime"].fillna("unknown")
    return out


def _breakdown(df: pd.DataFrame, group_col: str, min_n: int) -> dict:
    summary = summarize(df, group_col)
    summary["sufficient_sample"] = summary["trades"] >= min_n
    significance = cluster_significance(df, group_col, min_n=min_n)
    return {
        "summary": summary.to_dict("records"),
        "significant_vs_rest": significance.to_dict("records"),
    }


def _latest_report_fingerprint() -> tuple[int, object] | None:
    connection = conn()
    try:
        row = connection.execute(
            "SELECT sample_count, evaluated_through FROM research_diagnostics "
            "ORDER BY generated_at DESC LIMIT 1"
        ).fetchone()
        return (row[0], row[1]) if row is not None else None
    finally:
        connection.close()


def _persist_report(run_id: str, generated_at: datetime, sample_count: int, evaluated_through, report: dict) -> None:
    connection = conn()
    try:
        connection.execute(
            "INSERT INTO research_diagnostics VALUES (?, ?, ?, ?, ?)",
            [run_id, generated_at, sample_count, evaluated_through, json.dumps(report, indent=2, default=str)],
        )
    finally:
        connection.close()


def build_diagnostics_report(*, min_sample_size: int = MIN_SAMPLE_SIZE, force: bool = False) -> dict:
    """Build (and persist, as a new row) an aggregate diagnostics report over
    every evaluated paper observation accumulated so far.

    Idempotent: if the evaluated population is identical to the one behind the
    most recent report (same count, same latest evaluated_at), this is a
    no-op and returns {"skipped": True, ...} without writing a new row.
    Pass force=True to bypass that check (e.g. after a code/threshold change
    to the diagnostics logic itself, where the same data should be re-reported).
    """
    ensure_diagnostics_table()
    df = _load_evaluated_population()
    sample_count = len(df)
    evaluated_through = df["evaluated_at"].max() if sample_count else None

    if not force:
        fingerprint = _latest_report_fingerprint()
        if fingerprint is not None and fingerprint == (sample_count, evaluated_through):
            return {
                "skipped": True,
                "reason": "no new evaluated observations since the last report",
                "sample_count": sample_count,
            }

    run_id = f"diag-{uuid.uuid4().hex}"
    generated_at = datetime.now(timezone.utc)

    if sample_count == 0:
        overall = {**_empty_metrics(), "sufficient_sample": False}
        breakdowns = {dim: {"summary": [], "significant_vs_rest": []} for dim in _BREAKDOWN_DIMENSIONS}
    else:
        enriched = _add_derived_dimensions(df)
        overall_row = summarize(enriched).to_dict("records")[0]
        overall = {**overall_row, "sufficient_sample": sample_count >= min_sample_size}
        breakdowns = {dim: _breakdown(enriched, dim, min_sample_size) for dim in _BREAKDOWN_DIMENSIONS}

    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "sample_count": sample_count,
        "min_sample_size": min_sample_size,
        "overall": overall,
        "breakdowns": breakdowns,
    }
    _persist_report(run_id, generated_at, sample_count, evaluated_through, report)
    report["skipped"] = False
    return report


def _empty_metrics() -> dict:
    return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "profit_factor": None,
            "expectancy_r": 0.0, "total_r": 0.0, "net_points": 0.0, "max_drawdown_points": 0.0}
