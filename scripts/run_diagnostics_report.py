from __future__ import annotations

import argparse
import json

from intraday_engine.research.paper_diagnostics import MIN_SAMPLE_SIZE, build_diagnostics_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build (and persist, as a new run) an aggregate diagnostics report "
        "over accumulated evaluated paper observations"
    )
    parser.add_argument("--min-sample-size", type=int, default=MIN_SAMPLE_SIZE)
    parser.add_argument("--force", action="store_true", help="Persist a new report even if nothing changed")
    args = parser.parse_args()

    report = build_diagnostics_report(min_sample_size=args.min_sample_size, force=args.force)
    if report.get("skipped"):
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Diagnostics run_id={report['run_id']} sample_count={report['sample_count']} "
              f"sufficient_sample_overall={report['overall']['sufficient_sample']}")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
