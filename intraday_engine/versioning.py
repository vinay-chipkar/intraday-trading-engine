"""Provenance identifiers stamped onto new research observations.

These are manually-maintained version strings, bumped by a human whenever
the corresponding logic changes meaningfully -- not derived automatically,
since an automatic hash would change on every unrelated refactor and be
useless as a "did the logic I care about change" signal. Historical rows
recorded before this module existed are left untouched (NULL): do not
backfill a version onto data that was never actually stamped with one.
"""

from __future__ import annotations

import subprocess

# Bump when signals/engine.py's scoring weights, thresholds, or blockers change.
STRATEGY_VERSION = "1.0.0"

# Bump when strategy/point_in_time.py or technical/* feature computation changes.
FEATURE_ENGINE_VERSION = "1.0.0"

# Bump when the T+1-entry / same-bar-stop-wins / MAE-MFE execution assumptions
# in backtest/engine.py or research/paper_outcomes.py change.
# 1.1.0: paper_outcomes.py::evaluate_trade's mae_points used to be signed
# (negative for an adverse move); it now stores a non-negative magnitude,
# matching backtest/engine.py's convention exactly (both were already
# non-negative for mfe_points -- only mae_points disagreed). Rows stamped
# "1.0.0" (or NULL, for rows predating this column) used the old signed
# convention and were deliberately left un-rewritten -- do not average or
# otherwise combine their mae_points with "1.1.0"+ rows without accounting
# for the sign difference.
EXECUTION_MODEL_VERSION = "1.1.0"

# Bump when research/paper_learning.py::_failure_class's classification logic
# changes. Unlike the versions above (stamped once at generation time and
# never revisited), this one also drives a live rebuild: any
# paper_failure_analysis row whose diagnostics_version doesn't match this
# constant gets its failure_class recomputed from its own already-stored raw
# columns (outcome/side/signal_reasons/signal_blockers) -- see
# rebuild_stale_failure_classifications(). 2.0.0 reflects the fix that
# replaced a substring match on "VWAP" (true for nearly every trade
# regardless of any real conflict) with an actual side-vs-VWAP-reason check.
FAILURE_CLASSIFIER_VERSION = "2.0.0"


def get_code_commit() -> str | None:
    """Best-effort short git commit hash for the running checkout.

    Never raises: provenance is a nice-to-have, not something an observation
    should fail over. Returns None if git or repo history isn't available
    (e.g. a tarball deploy with no .git directory).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None
