from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from intraday_engine.research.paper_learning import (
    build_failure_analysis,
    learning_report,
    rebuild_stale_failure_classifications,
)
from intraday_engine.research.paper_outcomes import evaluate_pending, outcome_summary
from intraday_engine.storage.db import conn


def _export_table(table: str, output: Path) -> int:
    connection = conn()
    try:
        frame = connection.execute(f"SELECT * FROM {table} ORDER BY 1").df()
    finally:
        connection.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize paper research and export the audit journal")
    parser.add_argument("--output", default="research_data")
    parser.add_argument("--max-holding-bars", type=int, default=30)
    args = parser.parse_args()

    evaluation = evaluate_pending(max_holding_bars=args.max_holding_bars)
    new_failures = build_failure_analysis()
    # Idempotent: only touches rows whose failure_class was computed under an
    # older classifier version (e.g. the pre-fix substring-match VWAP-conflict
    # bug) -- never rewrites the underlying trade outcome, only the derived
    # diagnostic label.
    rebuilt_failures = rebuild_stale_failure_classifications()
    summary = outcome_summary()
    report = learning_report()

    root = Path(args.output)
    observations = _export_table("paper_observations", root / "paper_observations.csv")
    outcomes = _export_table("paper_outcomes", root / "paper_outcomes.csv")
    failures = _export_table("paper_failure_analysis", root / "paper_failure_analysis.csv")
    (root / "learning_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("===== PAPER CLOSE =====")
    print(json.dumps({
        "evaluation": evaluation,
        "new_failure_rows": new_failures,
        "rebuilt_stale_diagnostic_rows": rebuilt_failures,
        "summary": summary,
        "exported": {"observations": observations, "outcomes": outcomes, "failures": failures},
        "output": str(root),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
