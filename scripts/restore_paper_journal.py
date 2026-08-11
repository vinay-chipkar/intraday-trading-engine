from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from intraday_engine.research.paper_learning import ensure_learning_table
from intraday_engine.research.paper_observer import ensure_observation_table
from intraday_engine.research.paper_outcomes import ensure_outcome_table
from intraday_engine.storage.db import conn


def _restore(table: str, path: Path) -> int:
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    if frame.empty:
        return 0
    connection = conn()
    try:
        connection.register("journal_in", frame)
        connection.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM journal_in")
        return len(frame)
    finally:
        try:
            connection.unregister("journal_in")
        except Exception:
            pass
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore paper research journal into DuckDB")
    parser.add_argument("--input", default="research_data")
    args = parser.parse_args()
    root = Path(args.input)
    ensure_observation_table()
    ensure_outcome_table()
    ensure_learning_table()
    restored = {
        "observations": _restore("paper_observations", root / "paper_observations.csv"),
        "outcomes": _restore("paper_outcomes", root / "paper_outcomes.csv"),
        "failures": _restore("paper_failure_analysis", root / "paper_failure_analysis.csv"),
    }
    print(f"PAPER JOURNAL RESTORED {restored}")


if __name__ == "__main__":
    main()
