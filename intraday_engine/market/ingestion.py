from __future__ import annotations

from dataclasses import dataclass
import logging

from intraday_engine.market.candles import normalize_candles, quality_report
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import (
    get_instruments,
    insert_candles,
    latest_candle_timestamp,
)


LOGGER = logging.getLogger(__name__)

# A handful of symbol-level failures (a delisted ticker, one transient 500) is
# normal degradation and is tolerated as a warning. But the scanner's ranking
# is cross-sectional (percentiles across the universe) and the signal engine
# needs same-day bars -- if most of the universe failed to ingest, the result
# is not a smaller-but-usable dataset, it's an unusable one, and must not be
# treated as a successful run.
MIN_INGESTION_SUCCESS_RATIO = 0.5


class IngestionFailure(RuntimeError):
    """Raised when an ingestion batch is too degraded for downstream use."""


def assess_ingestion_results(results: list[IngestionResult]) -> None:
    """Raise IngestionFailure when an ingestion batch is unusable as a whole.

    Silent partial/total failure is exactly how this pipeline's staleness and
    empty-outcome incidents went undetected in the past: per-symbol errors
    were recorded but nothing ever turned "most of the universe failed" into a
    hard failure the caller (and CI) could see.
    """
    if not results:
        raise IngestionFailure("ingestion produced no results at all (empty symbol universe?)")
    failed = [result for result in results if result.error]
    success_ratio = (len(results) - len(failed)) / len(results)
    if success_ratio < MIN_INGESTION_SUCCESS_RATIO:
        failed_symbols = ", ".join(result.symbol for result in failed)
        raise IngestionFailure(
            f"ingestion failed for {len(failed)}/{len(results)} symbols "
            f"({success_ratio:.0%} succeeded, below the {MIN_INGESTION_SUCCESS_RATIO:.0%} "
            f"minimum): {failed_symbols}"
        )


@dataclass(frozen=True)
class IngestionResult:
    symbol: str
    rows_received: int
    rows_inserted: int
    last_timestamp: object | None
    quality: dict[str, int | bool]
    error: str | None = None


def ingest_symbol(
    api: UpstoxREST,
    *,
    symbol: str,
    instrument_key: str,
    interval: int = 1,
) -> IngestionResult:
    interval_key = f"{interval}m"
    try:
        frame = api.intraday_candles(
            instrument_key,
            unit="minutes",
            interval=interval,
        )
        report = quality_report(frame)
        normalized = normalize_candles(
            frame,
            instrument_key=instrument_key,
            symbol=symbol,
            interval=interval_key,
        )

        last_stored = latest_candle_timestamp(instrument_key, interval_key)
        if last_stored is not None:
            normalized = normalized[normalized["timestamp"] > last_stored].copy()

        inserted = insert_candles(normalized)
        last_timestamp = normalized["timestamp"].max() if not normalized.empty else last_stored
        return IngestionResult(
            symbol=symbol,
            rows_received=len(frame),
            rows_inserted=inserted,
            last_timestamp=last_timestamp,
            quality=report,
        )
    except Exception as exc:
        LOGGER.exception("Intraday ingestion failed for %s", symbol)
        return IngestionResult(
            symbol=symbol,
            rows_received=0,
            rows_inserted=0,
            last_timestamp=None,
            quality={},
            error=str(exc),
        )


def ingest_symbols(symbols: list[str] | None = None, interval: int = 1) -> list[IngestionResult]:
    api = UpstoxREST()
    instruments = get_instruments(symbols)
    if instruments.empty:
        raise LookupError("instrument_master is empty. Run scripts/sync_instruments.py first.")

    results: list[IngestionResult] = []
    for row in instruments.itertuples(index=False):
        results.append(
            ingest_symbol(
                api,
                symbol=row.symbol,
                instrument_key=row.instrument_key,
                interval=interval,
            )
        )
    return results
