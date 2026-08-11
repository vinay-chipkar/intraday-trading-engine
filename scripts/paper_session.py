from __future__ import annotations

import argparse
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from config.settings import settings
from intraday_engine.market.ingestion import ingest_symbols
from intraday_engine.research.daily_cycle import run_daily_cycle
from intraday_engine.research.paper_observer import observe_once
from intraday_engine.research.paper_outcomes import evaluate_pending, outcome_summary

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)


def _now() -> datetime:
    return datetime.now(IST)


def _parse_clock(value: str) -> dt_time:
    try:
        hour, minute = (int(part) for part in value.split(":", 1))
        return dt_time(hour, minute)
    except (TypeError, ValueError) as exc:
        raise ValueError("--until must use HH:MM") from exc


def _refresh_candles() -> None:
    results = ingest_symbols(interval=settings.candle_interval)
    failed = [result for result in results if result.error]
    inserted = sum(result.rows_inserted for result in results)
    print(
        f"DATA TICK {len(results) - len(failed)}/{len(results)} symbols ok "
        f"new_candles={inserted}"
    )
    for result in failed:
        print(f"DATA WARNING {result.symbol}: {result.error}")


def _run_once(limit: int, min_score: float, max_holding_bars: int) -> None:
    _refresh_candles()
    observed = observe_once(limit=limit, min_score=min_score)
    evaluation = evaluate_pending(max_holding_bars=max_holding_bars)
    summary = outcome_summary()
    print(
        f"PAPER TICK observed={len(observed)} "
        f"evaluated={evaluation['evaluated']} waiting={evaluation['waiting']} "
        f"total={summary['evaluated']} wins={summary['wins']} "
        f"losses={summary['losses']} avg_r={summary['avg_r']:.4f}"
    )


def _bootstrap(limit: int) -> None:
    print("PAPER BOOTSTRAP starting")
    report = run_daily_cycle(mode="PAPER", limit=limit)
    print(
        f"PAPER BOOTSTRAP complete universe={report['universe_size']} "
        f"candidates={report['candidate_count']}"
    )


def _finalize(max_holding_bars: int) -> None:
    print("PAPER WINDOW CLOSED; final paper evaluation")
    evaluation = evaluate_pending(max_holding_bars=max_holding_bars)
    summary = outcome_summary()
    print(f"FINAL evaluation={evaluation} summary={summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper observer/evaluator repeatedly during NSE market hours"
    )
    parser.add_argument("--interval", type=int, default=5, help="Minutes between paper ticks")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--max-holding-bars", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Run one tick and exit")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip startup instrument/premarket/data bootstrap")
    parser.add_argument("--until", default=None, help="Stop this runner window at HH:MM IST")
    args = parser.parse_args()

    if args.interval < 1:
        raise ValueError("--interval must be >= 1")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.min_score < 0:
        raise ValueError("--min-score must be >= 0")
    if args.max_holding_bars < 1:
        raise ValueError("--max-holding-bars must be >= 1")
    until = _parse_clock(args.until) if args.until else MARKET_CLOSE
    if until <= MARKET_OPEN or until > MARKET_CLOSE:
        raise ValueError("--until must be after 09:15 and no later than 15:30")

    if args.once:
        if not args.no_bootstrap:
            _bootstrap(args.limit)
        _run_once(args.limit, args.min_score, args.max_holding_bars)
        return

    print(
        f"PAPER SESSION started interval={args.interval}m "
        f"window={MARKET_OPEN.strftime('%H:%M')}-{until.strftime('%H:%M')} IST"
    )
    if not args.no_bootstrap:
        _bootstrap(args.limit)

    while True:
        now = _now()
        current = now.time().replace(tzinfo=None)

        if current < MARKET_OPEN:
            wait_seconds = max(1, int((datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST) - now).total_seconds()))
            print(f"Before market open; sleeping {wait_seconds}s")
            time.sleep(min(wait_seconds, 60))
            continue

        if current >= until:
            _finalize(args.max_holding_bars)
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
