from __future__ import annotations

import argparse
import json

from intraday_engine.research.paper_observer import observe_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a paper-trading observation snapshot; never places orders"
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=60.0)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.min_score < 0:
        raise ValueError("--min-score must be >= 0")

    rows = observe_once(limit=args.limit, min_score=args.min_score)
    print("===== PAPER OBSERVATIONS =====")
    for row in rows:
        print(
            f"#{row['scanner_rank']} {row['symbol']:<12} "
            f"candidate={row['candidate_score']:.2f} "
            f"signal={row['signal_action']:<9} "
            f"signal_score={row['signal_score']:.2f} "
            f"confidence={row['confidence']:.2f} "
            f"status={row['status']}"
        )
        if row["signal_action"] in {"BUY", "SELL"}:
            print(
                f"    entry={row['entry_price']:.4f} "
                f"stop={row['stop_loss']:.4f} "
                f"target={row['target']:.4f}"
            )
        print(f"    reasons={row['signal_reasons']}")
        print(f"    blockers={row['signal_blockers']}")

    print(json.dumps({"observations": len(rows), "mode": "PAPER"}))


if __name__ == "__main__":
    main()
