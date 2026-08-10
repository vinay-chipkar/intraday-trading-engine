from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging
import uuid

import pandas as pd

from intraday_engine.market.candles import normalize_candles
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import get_instruments, insert_candles, conn


LOGGER = logging.getLogger(__name__)
MAX_MINUTE_WINDOW_DAYS = 30


@dataclass(frozen=True)
class BackfillResult:
    symbol: str
    requested_rows: int
    inserted_rows: int
    windows: int


def date_windows(start: date, end: date, max_days: int = MAX_MINUTE_WINDOW_DAYS):
    """Yield inclusive date windows small enough for Upstox minute-data limits."""
    if start > end:
        raise ValueError("start date must be on or before end date")
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=max_days - 1), end)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def backfill_symbol(
    api: UpstoxREST,
    *,
    symbol: str,
    instrument_key: str,
    start: date,
    end: date,
    interval: int = 1,
) -> BackfillResult:
    requested = 0
    inserted = 0
    windows = 0

    for window_start, window_end in date_windows(start, end):
        windows += 1
        LOGGER.info("Backfilling %s: %s -> %s", symbol, window_start, window_end)
        frame = api.historical_candles(
            instrument_key,
            unit="minutes",
            interval=interval,
            to_date=window_end,
            from_date=window_start,
        )
        requested += len(frame)
        normalized = normalize_candles(
            frame,
            instrument_key=instrument_key,
            symbol=symbol,
            interval=f"{interval}m",
        )
        inserted += insert_candles(normalized)

    return BackfillResult(symbol, requested, inserted, windows)


def backfill(
    symbols: list[str],
    *,
    start: date,
    end: date,
    interval: int = 1,
) -> list[BackfillResult]:
    api = UpstoxREST()
    instruments = get_instruments(symbols)
    available = {row.symbol.upper(): row.instrument_key for row in instruments.itertuples()}
    missing = [symbol.upper() for symbol in symbols if symbol.upper() not in available]
    if missing:
        raise LookupError(
            "Missing instruments. Run scripts/sync_instruments.py first: "
            + ", ".join(missing)
        )

    results: list[BackfillResult] = []
    for symbol in symbols:
        results.append(
            backfill_symbol(
                api,
                symbol=symbol.upper(),
                instrument_key=available[symbol.upper()],
                start=start,
                end=end,
                interval=interval,
            )
        )
    return results


def record_ingestion_run(
    *,
    mode: str,
    interval: str,
    requested_symbols: int,
    successful_symbols: int,
    rows_received: int,
    rows_inserted: int,
    status: str,
    error: str | None = None,
    started_at=None,
    finished_at=None,
) -> None:
    run_id = str(uuid.uuid4())
    started_at = started_at or pd.Timestamp.now(tz="Asia/Kolkata")
    finished_at = finished_at or pd.Timestamp.now(tz="Asia/Kolkata")
    connection = conn()
    try:
        connection.execute(
            """
            INSERT INTO ingestion_runs
            (run_id, started_at, finished_at, mode, interval,
             requested_symbols, successful_symbols, rows_received,
             rows_inserted, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, started_at, finished_at, mode, interval,
                requested_symbols, successful_symbols, rows_received,
                rows_inserted, status, error,
            ],
        )
    finally:
        connection.close()
