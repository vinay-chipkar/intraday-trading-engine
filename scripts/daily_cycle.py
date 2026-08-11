from __future__ import annotations

import argparse
import json

from intraday_engine.research.daily_cycle import run_daily_cycle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the research-first daily paper-trading preparation cycle"
    )
    parser.add_argument(
        "--mode",
        choices=["PAPER"],
        default="PAPER",
        help="Trading mode. LIVE is intentionally unavailable during research.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-premarket", action="store_true")
    args = parser.parse_args()

    report = run_daily_cycle(
        mode=args.mode,
        limit=args.limit,
        skip_sync=args.skip_sync,
        skip_ingest=args.skip_ingest,
        skip_premarket=args.skip_premarket,
    )

    print("\n===== DAILY PAPER RESEARCH =====")
    print(f"run_id={report['run_id']}")
    print(f"mode={report['mode']}")
    print(f"trading_date={report['trading_date']}")
    print(f"universe={report['universe_size']}")
    print(f"market_regime={report['market_regime']}")
    print(f"market_score={report['market_score']}")
    print(f"candidates={report['candidate_count']}")
    for row in report["candidates"]:
        print(
            f"#{row['rank']} {row['symbol']} score={row['candidate_score']:.2f} "
            f"change={row['change_pct']:.2f}% rvol={row['relative_volume']:.2f}x "
            f"news={row.get('news_score', 0.0):+.2f}"
        )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print("\nNo live orders are permitted by this command.")
    print(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
