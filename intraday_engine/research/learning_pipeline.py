"""Turn completed (evaluated) paper observations/outcomes into point-in-time
feature snapshots and training labels, automatically and idempotently.

Causality contract, enforced by construction, not by convention:

- The feature snapshot for an observation is, whenever available, the exact
  point-in-time feature vector `paper_observer.py::_signal_for_symbol`
  captured and stored (`paper_observations.decision_features`) at the moment
  it built the live signal -- not a later recomputation. This matters
  because feature_engine.py's re-enrichment is an independent implementation
  from strategy/point_in_time.py's live path (different support/resistance/
  trend definitions), and because candles can be revised after the signal
  was generated (market/ingestion.py::upsert_candles) -- recomputing later
  would silently drift from what the signal actually saw. Observations that
  predate decision_features capture fall back to the old approach: features
  re-derived from candles at or before the observation's own `bar_time`,
  never anything after.
- The training label is *not* re-derived from raw candles at all. It is
  copied directly from the observation's `paper_outcomes` row, which was
  itself computed by `paper_outcomes.py::evaluate_trade` -- already verified
  (see tests/test_paper_outcomes.py and this session's history) to use only
  bars strictly after the signal bar. Re-deriving a second, independent label
  from `ml/labels.py` here would risk two implementations silently
  disagreeing; reusing the one source of truth for "what happened after this
  signal" avoids that class of bug entirely.
- An observation is only processed once it is EVALUATED (i.e. has a
  paper_outcomes row) -- a still-PENDING observation has no known outcome yet
  and therefore cannot be labeled without guessing at the future.
"""

from __future__ import annotations

import json

import pandas as pd

from intraday_engine.storage.db import (
    FEATURE_SNAPSHOT_COLUMNS,
    TRAINING_LABEL_COLUMNS,
    conn,
    insert_df,
)
from intraday_engine.technical.feature_engine import latest_feature_snapshot

_TARGET_OUTCOMES = {"TARGET", "TARGET_GAP"}


def _load_unprocessed_evaluated_observations() -> pd.DataFrame:
    """Every EVALUATED paper observation not yet turned into a feature snapshot."""
    connection = conn()
    try:
        return connection.execute(
            """
            SELECT o.observation_id, o.symbol, o.instrument_key, o.bar_time, o.decision_features,
                   p.entry_price, p.target AS target_price, p.stop_loss AS stop_price,
                   p.outcome, p.mfe_points, p.mae_points
            FROM paper_observations o
            JOIN paper_outcomes p USING (observation_id)
            WHERE o.status = 'EVALUATED'
              AND NOT EXISTS (
                  SELECT 1 FROM feature_snapshots f WHERE f.observation_id = o.observation_id
              )
            ORDER BY o.bar_time
            """
        ).df()
    finally:
        connection.close()


def _snapshot_from_decision_features(decision_features_json: str, *, observation_id: str) -> dict:
    """Rebuild the exact captured decision-time snapshot dict from its stored
    JSON, restoring the datetime/date types json.dumps(default=str) flattened
    to plain strings."""
    parsed = json.loads(decision_features_json)
    parsed["event_time"] = pd.Timestamp(parsed["event_time"]).to_pydatetime()
    parsed["trading_date"] = pd.Timestamp(parsed["trading_date"]).date()
    parsed["observation_id"] = observation_id
    return parsed


def _causal_candles(instrument_key: str, as_of) -> pd.DataFrame:
    """Every 1m candle for `instrument_key` at or before `as_of` -- never
    after. Filtered by instrument_key, not symbol -- see
    paper_observer.py::_load_bars for why a symbol-only query risks mixing
    histories across a remapped instrument_key (relisting/rename)."""
    connection = conn()
    try:
        return connection.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE instrument_key = ? AND interval = '1m' AND timestamp <= ?
            ORDER BY timestamp
            """,
            [instrument_key, as_of],
        ).df()
    finally:
        connection.close()


def build_feature_snapshots_and_labels(*, horizon_minutes: int = 30) -> dict:
    """Process every evaluated-but-unprocessed paper observation.

    Idempotent: an observation already represented in feature_snapshots is
    never reprocessed, so this can run on every workflow tick with no
    duplication and no manual bookkeeping.
    """
    pending = _load_unprocessed_evaluated_observations()

    snapshot_rows: list[dict] = []
    label_rows: list[dict] = []
    skipped_no_candles: list[str] = []
    from_decision_features = 0
    from_recomputed_candles = 0

    for row in pending.itertuples(index=False):
        decision_features = getattr(row, "decision_features", None)
        if decision_features:
            # Exact decision-time capture exists -- use it directly. Never
            # recompute when this is available: recomputation is an
            # independent implementation (feature_engine.py vs.
            # strategy/point_in_time.py) and is vulnerable to candles having
            # been revised after the signal was generated.
            snapshot = _snapshot_from_decision_features(decision_features, observation_id=row.observation_id)
            from_decision_features += 1
        else:
            # Historical observation predating decision_features capture --
            # fall back to the old recompute-from-candles approximation.
            bars = _causal_candles(row.instrument_key, row.bar_time)
            if bars.empty or pd.Timestamp(bars["timestamp"].iloc[-1]) != pd.Timestamp(row.bar_time):
                # No candle at all, or (defensively) the causal query's last
                # row isn't exactly the observation's own bar -- refuse to guess.
                skipped_no_candles.append(row.observation_id)
                continue
            snapshot = latest_feature_snapshot(bars, symbol=row.symbol, instrument_key=row.instrument_key)
            snapshot["observation_id"] = row.observation_id
            from_recomputed_candles += 1

        snapshot_rows.append({column: snapshot.get(column) for column in FEATURE_SNAPSHOT_COLUMNS})

        target_hit_first = row.outcome in _TARGET_OUTCOMES
        label_rows.append({
            "event_time": row.bar_time,
            "symbol": row.symbol,
            "horizon_minutes": horizon_minutes,
            "entry_price": row.entry_price,
            "target_price": row.target_price,
            "stop_price": row.stop_price,
            "target_hit_first": target_hit_first,
            "max_favorable_excursion": row.mfe_points,
            "max_adverse_excursion": row.mae_points,
            "label": int(target_hit_first),
            "observation_id": row.observation_id,
        })

    if snapshot_rows:
        insert_df("feature_snapshots", pd.DataFrame(snapshot_rows, columns=FEATURE_SNAPSHOT_COLUMNS))
    if label_rows:
        insert_df("training_labels", pd.DataFrame(label_rows, columns=TRAINING_LABEL_COLUMNS))

    return {
        "evaluated_pending": len(pending),
        "feature_snapshots_written": len(snapshot_rows),
        "training_labels_written": len(label_rows),
        "skipped_no_candles": len(skipped_no_candles),
        "from_decision_features": from_decision_features,
        "from_recomputed_candles": from_recomputed_candles,
    }
