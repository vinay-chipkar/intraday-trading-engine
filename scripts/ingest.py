from __future__ import annotations

import argparse
import logging

from intraday_engine.market.ingestion import ingest_symbols
from intraday_engine.market.universe import load_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally ingest today's Upstox candles.")
    parser.add_argument("--symbol", action="append", dest="symbols", help="NSE symbol; repeat for multiple symbols")
    parser.add_argument("--interval", type=int, default=1, choices=range(1, 301))
    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols] if args.symbols else load_symbols()
    results = ingest_symbols(symbols=symbols, interval=args.interval)
    failures = 0
    for result in results:
        print(f"{result.symbol}: received={result.rows_received} written={result.rows_written} last={result.last_timestamp}")
        print(f"  QUALITY: {result.quality}")
        if result.error:
            failures += 1
            print(f"  ERROR: {result.error}")
    if failures:
        raise SystemExit(f"{failures} symbol(s) failed ingestion")


if __name__ == "__main__":
    main()
