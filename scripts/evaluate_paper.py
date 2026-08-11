from __future__ import annotations

import argparse
import json

from intraday_engine.research.paper_outcomes import evaluate_pending, outcome_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate completed paper-trading signals against future 1m candles"
    )
    parser.add_argument("--max-holding-bars", type=int, default=30)
    args = parser.parse_args()
    if args.max_holding_bars < 1:
        raise ValueError("--max-holding-bars must be >= 1")

    result = evaluate_pending(max_holding_bars=args.max_holding_bars)
    summary = outcome_summary()

    print("===== PAPER OUTCOME EVALUATION =====")
    print(f"pending_seen={result['pending']}")
    print(f"newly_evaluated={result['evaluated']}")
    print(f"waiting_for_future_bars={result['waiting']}")
    print(f"total_evaluated={summary['evaluated']}")
    print(f"wins={summary['wins']}")
    print(f"losses={summary['losses']}")
    print(f"win_rate={summary['win_rate']:.4f}")
    print(f"avg_r={summary['avg_r']:.4f}")
    print(f"profit_factor={summary['profit_factor']}")
    print(json.dumps({**result, **summary}, default=str))


if __name__ == "__main__":
    main()
