from __future__ import annotations

import argparse
import json

from intraday_engine.research.learning_pipeline import build_feature_snapshots_and_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn evaluated paper observations/outcomes into point-in-time "
        "feature snapshots and training labels"
    )
    parser.add_argument("--horizon-minutes", type=int, default=30)
    args = parser.parse_args()

    summary = build_feature_snapshots_and_labels(horizon_minutes=args.horizon_minutes)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
