from __future__ import annotations

import argparse
import json
import sys

from intraday_engine.research.monitoring import PipelineUnhealthy, build_monitoring_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build (and persist, as a new run) an operational health report over "
        "accumulated ingestion/paper-trading history"
    )
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--sample-growth-lookback-days", type=int, default=90)
    parser.add_argument("--consecutive-failure-threshold", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Persist a new report even if nothing changed")
    parser.add_argument(
        "--no-fail-on-unhealthy",
        action="store_true",
        help="Report an unhealthy pipeline without failing the process (default: fail loudly)",
    )
    args = parser.parse_args()

    try:
        report = build_monitoring_report(
            lookback_days=args.lookback_days,
            sample_growth_lookback_days=args.sample_growth_lookback_days,
            consecutive_failure_threshold=args.consecutive_failure_threshold,
            force=args.force,
            fail_on_unhealthy=not args.no_fail_on_unhealthy,
        )
    except PipelineUnhealthy as exc:
        print(f"PIPELINE UNHEALTHY: {exc}", file=sys.stderr)
        raise

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
