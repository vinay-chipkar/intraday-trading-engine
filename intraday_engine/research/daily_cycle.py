from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config.settings import settings
from intraday_engine.market.ingestion import assess_ingestion_results, ingest_symbols
from intraday_engine.market.universe import sync_instruments
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.scanner.service import ScannerConfig, scan_top10
from intraday_engine.storage.db import (
    get_instruments,
    insert_research_run,
    latest_market_context,
)
from scripts.premarket import main as capture_premarket


IST = ZoneInfo("Asia/Kolkata")


def run_daily_cycle(
    *,
    mode: str = "PAPER",
    limit: int | None = None,
    skip_sync: bool = False,
    skip_ingest: bool = False,
    skip_premarket: bool = False,
) -> dict:
    """Run the safe daily research/paper preparation cycle.

    This function deliberately stops at candidate generation. It never places
    live orders and refuses any mode other than PAPER.
    """
    if mode.upper() != "PAPER":
        raise RuntimeError("Only PAPER mode is permitted by the research cycle")

    started = datetime.now(timezone.utc)
    run_id = f"research-{started.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    trading_date = started.astimezone(IST).date()
    errors: list[str] = []

    try:
        if not skip_sync:
            synced = sync_instruments()
            print(f"Instrument sync: {len(synced)} resolved")

        instruments = get_instruments()
        universe_size = len(instruments)
        if universe_size == 0:
            raise LookupError("instrument_master is empty")

        if not skip_premarket:
            capture_premarket()

        if not skip_ingest:
            results = ingest_symbols(interval=settings.candle_interval)
            assess_ingestion_results(results)
            failed = [result for result in results if result.error]
            inserted = sum(result.rows_inserted for result in results)
            print(
                f"Intraday ingestion: {len(results) - len(failed)}/{len(results)} symbols ok, "
                f"{inserted} new candles"
            )
            errors.extend(f"{result.symbol}: {result.error}" for result in failed)

        context = latest_market_context() or {}
        candidates = scan_top10(
            UpstoxREST(),
            ScannerConfig(
                limit=limit or settings.top_n,
                news_lookback_hours=settings.news_lookback_hours,
            ),
            trading_date=trading_date,
        )

        report = {
            "run_id": run_id,
            "mode": "PAPER",
            "trading_date": trading_date.isoformat(),
            "universe_size": universe_size,
            "candidate_count": len(candidates),
            "market_regime": context.get("regime"),
            "market_score": context.get("score"),
            "candidates": candidates,
            "warnings": errors,
        }
        status = "WARNING" if errors else "SUCCESS"
        finished = datetime.now(timezone.utc)
        insert_research_run(
            {
                "run_id": run_id,
                "started_at": started,
                "finished_at": finished,
                "trading_date": trading_date,
                "mode": "PAPER",
                "status": status,
                "universe_size": universe_size,
                "candidates_count": len(candidates),
                "market_regime": context.get("regime"),
                "market_score": context.get("score"),
                "report_json": json.dumps(report, default=str),
                "error": None,
            }
        )
        return report
    except Exception as exc:
        finished = datetime.now(timezone.utc)
        insert_research_run(
            {
                "run_id": run_id,
                "started_at": started,
                "finished_at": finished,
                "trading_date": trading_date,
                "mode": "PAPER",
                "status": "FAILED",
                "universe_size": len(get_instruments()),
                "candidates_count": 0,
                "market_regime": None,
                "market_score": None,
                "report_json": json.dumps({"run_id": run_id, "warnings": errors}),
                "error": str(exc),
            }
        )
        raise
