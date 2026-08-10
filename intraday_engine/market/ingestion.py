from __future__ import annotations

from dataclasses import dataclass
import logging

from intraday_engine.market.candles import normalize_candles, quality_report
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import get_instruments, insert_candles, latest_candle_timestamp

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    symbol: str
    rows_received: int
    rows_inserted: int
    last_timestamp: object | None
    quality: dict[str, int | bool]
    error: str | None = None


def ingest_symbol(api: UpstoxREST, *, symbol: str, instrument_key: str, interval: int = 1) -> IngestionResult:
    interval_key = f"{interval}m"
    try:
        frame = api.intraday_candles(instrument_key, unit="minutes", interval=interval)
        report = quality_report(frame)
        normalized = normalize_candles(frame, instrument_key=instrument_key, symbol=symbol, interval=interval_key)
        last_stored = latest_candle_timestamp(instrument_key, interval_key)
        if last_stored is not None:
            normalized = normalized[normalized["timestamp"] > last_stored].copy()
        inserted = insert_candles(normalized)
        last_timestamp = normalized["timestamp"].max() if not normalized.empty else last_stored
        return IngestionResult(symbol, len(frame), inserted, last_timestamp, report)
    except Exception as exc:
        LOGGER.exception("Intraday ingestion failed for %s", symbol)
        return IngestionResult(symbol, 0, 0, None, {}, str(exc))


def ingest_symbols(symbols: list[str] | None = None, interval: int = 1) -> list[IngestionResult]:
    api = UpstoxREST()
    instruments = get_instruments(symbols)
    if instruments.empty:
        raise LookupError("instrument_master is empty. Run scripts/sync_instruments.py first.")
    return [
        ingest_symbol(api, symbol=row.symbol, instrument_key=row.instrument_key, interval=interval)
        for row in instruments.itertuples(index=False)
    ]
