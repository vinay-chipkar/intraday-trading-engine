from __future__ import annotations

import argparse
from datetime import date, timedelta

from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import get_instruments, insert_candles
from intraday_engine.market.candles import normalize_candles


def backfill(days: int = 30) -> int:
    if days < 1:
        raise ValueError("days must be >= 1")

    api = UpstoxREST()
    instruments = get_instruments()
    if instruments.empty:
        raise LookupError("instrument_master is empty. Run scripts/sync_instruments.py first.")

    # Upstox historical 1m data is requested as a bounded date range. Keep the
    # default at 30 days, which is enough to build the scanner's 20-day lookback.
    to_date = date.today()
    from_date = to_date - timedelta(days=days - 1)
    total_received = 0
    total_inserted = 0

    print(f"Backfill window: {from_date} -> {to_date}")
    print(f"Universe: {len(instruments)} symbols")

    for row in instruments.itertuples(index=False):
        symbol = str(row.symbol)
        key = str(row.instrument_key)
        try:
            frame = api.historical_candles(
                key,
                unit="minutes",
                interval=1,
                to_date=to_date,
                from_date=from_date,
            )
            total_received += len(frame)
            if frame.empty:
                print(f"WARN {symbol}: no historical candles returned")
                continue

            normalized = normalize_candles(
                frame,
                instrument_key=key,
                symbol=symbol,
                interval="1m",
            )
            inserted = insert_candles(normalized)
            total_inserted += inserted
            print(f"{symbol}: received={len(frame)} inserted={inserted}")
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")

    print(f"Backfill complete: received={total_received} inserted={total_inserted}")
    return total_inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill 1m history for the full scanner universe")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    backfill(args.days)


if __name__ == "__main__":
    main()
