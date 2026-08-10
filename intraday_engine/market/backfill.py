from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging

from intraday_engine.market.candles import normalize_candles
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import get_instruments, insert_candles

LOGGER = logging.getLogger(__name__)
MAX_MINUTE_WINDOW_DAYS = 30


@dataclass(frozen=True)
class BackfillResult:
    symbol: str
    requested_rows: int
    inserted_rows: int
    windows: int


def date_windows(start: date, end: date, max_days: int = MAX_MINUTE_WINDOW_DAYS):
    if start > end:
        raise ValueError("start date must be on or before end date")
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=max_days - 1), end)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def backfill_symbol(api: UpstoxREST, *, symbol: str, instrument_key: str, start: date, end: date, interval: int = 1) -> BackfillResult:
    requested = inserted = windows = 0
    for window_start, window_end in date_windows(start, end):
        windows += 1
        LOGGER.info("Backfilling %s: %s -> %s", symbol, window_start, window_end)
        frame = api.historical_candles(instrument_key, unit="minutes", interval=interval, to_date=window_end, from_date=window_start)
        requested += len(frame)
        normalized = normalize_candles(frame, instrument_key=instrument_key, symbol=symbol, interval=f"{interval}m")
        inserted += insert_candles(normalized)
    return BackfillResult(symbol, requested, inserted, windows)


def backfill(symbols: list[str], *, start: date, end: date, interval: int = 1) -> list[BackfillResult]:
    api = UpstoxREST()
    instruments = get_instruments(symbols)
    available = {row.symbol.upper(): row.instrument_key for row in instruments.itertuples()}
    missing = [symbol.upper() for symbol in symbols if symbol.upper() not in available]
    if missing:
        raise LookupError("Missing instruments. Run scripts/sync_instruments.py first: " + ", ".join(missing))
    return [
        backfill_symbol(api, symbol=symbol.upper(), instrument_key=available[symbol.upper()], start=start, end=end, interval=interval)
        for symbol in symbols
    ]
