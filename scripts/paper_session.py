from __future__ import annotations

import argparse
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from intraday_engine.research.paper_observer import observe_once
from intraday_engine.research.paper_outcomes import evaluate_pending, outcome_summary

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)


def _now() -> datetime:
    return datetime.now(IST)


def _run_once(limit: int, min_score: float, max_holding_bars: int) -> None:
    observed = observe_once(limit=limit, min_score=min_score)
    evaluation = evaluate_pending(max_holding_bars=max_holding_bars)
    summary = outcome_summary()
    print(
        f"PAPER TICK observed={len(observed)} "
        f"evaluated={evaluation['evaluated']} waiting={evaluation['waiting']} "
        f"total={summary['evaluated']} wins={summary['wins']} "
        f"losses={summary['losses']} avg_r={summary['avg_r']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper observer/evaluator repeatedly during NSE market hours"
    )
    parser.add_argument("--interval", type=int, default=5, help="Minutes between paper ticks")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--max-holding-bars", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Run one tick and exit")
    args = parser.parse_args()

    if args.interval < 1:
        raise ValueError("--interval must be >= 1")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.min_score < 0:
        raise ValueError("--min-score must be >= 0")
    if args.max_holding_bars < 1:
        raise ValueError("--max-holding-bars must be >= 1")

    if args.once:
        _run_once(args.limit, args.min_score, args.max_holding_bars)
        return

    print(
        f"PAPER SESSION started interval={args.interval}m "
        f"window={MARKET_OPEN.strftime('%H:%M')}-{MARKET_CLOSE.strftime('%H:%M')} IST"
    )

    while True:
        now = _now()
        current = now.time().replace(tzinfo=None)

        if current < MARKET_OPEN:
            wait_seconds = max(1, int((datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST) - now).total_seconds()))
            print(f"Before market open; sleeping {wait_seconds}s")
            time.sleep(min(wait_seconds, 60))
            continue

        if current >= MARKET_CLOSE:
            print("NSE market session closed; final paper evaluation")
            evaluation = evaluate_pending(max_holding_bars=args.max_holding_bars)
            summary = outcome_summary()
            print(f"FINAL evaluation={evaluation} summary={summary}")
            return

        started = time.monotonic()
        try:
            _run_once(args.limit, args.min_score, args.max_holding_bars)
        except KeyboardInterrupt:
            print("PAPER SESSION stopped by user")
            return
        except Exception as exc:
            print(f"PAPER TICK ERROR: {type(exc).__name__}: {exc}")

        elapsed = time.monotonic() - started
        time.sleep(max(1, args.interval * 60 - elapsed))


if __name__ == "__main__":
    main()
