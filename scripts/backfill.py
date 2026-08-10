from __future__ import annotations

import argparse
from datetime import date, timedelta

from intraday_engine.market.backfill import backfill
from intraday_engine.market.universe import load_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Upstox minute candles into DuckDB.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", action="append", dest="symbols", help="NSE symbol; repeat for multiple symbols")
    group.add_argument("--universe", action="store_true", help="Backfill every symbol in config/universe.csv")
    parser.add_argument("--days", type=int, default=30, help="Calendar days ending on the selected end date")
    parser.add_argument("--start", type=date.fromisoformat, help="Explicit start date YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="Explicit end date YYYY-MM-DD")
    parser.add_argument("--interval", type=int, default=1, choices=range(1, 16), help="Minute interval 1-15")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.start) != bool(args.end):
        raise SystemExit("--start and --end must be supplied together")
    end = args.end or date.today()
    start = args.start or (end - timedelta(days=max(args.days, 1) - 1))
    symbols = [s.upper() for s in args.symbols] if args.symbols else load_symbols()
    for result in backfill(symbols, start=start, end=end, interval=args.interval):
        print(f"{result.symbol}: received={result.requested_rows} inserted={result.inserted_rows} windows={result.windows}")


if __name__ == "__main__":
    main()
