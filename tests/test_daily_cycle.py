import pandas as pd
import pytest

from intraday_engine.market.ingestion import IngestionFailure, IngestionResult
from intraday_engine.research import daily_cycle
from intraday_engine.research.daily_cycle import run_daily_cycle


def test_daily_cycle_refuses_live_mode():
    with pytest.raises(RuntimeError, match="Only PAPER mode"):
        run_daily_cycle(mode="LIVE")


def _ingestion_result(symbol: str, *, error: str | None = None) -> IngestionResult:
    # rows_received=0 now counts as an unhealthy result on its own (see
    # assess_ingestion_results), so a "successful" symbol here must carry a
    # realistic non-zero row count -- otherwise every symbol looks failed
    # regardless of whether `error` is set.
    return IngestionResult(
        symbol=symbol, rows_received=0 if error else 5, rows_inserted=0 if error else 5,
        last_timestamp=None, quality={}, error=error,
    )


def test_daily_cycle_fails_non_zero_and_records_failed_status_when_all_symbols_fail_ingestion(monkeypatch):
    # Requirement: "if all symbol ingestion fails, the daily cycle must fail
    # non-zero." Before this fix, a total ingestion failure only ever produced
    # a WARNING-status research_run and a normal return -- the CLI wrapper
    # (scripts/daily_cycle.py) would exit 0.
    recorded_runs = []
    monkeypatch.setattr(daily_cycle, "sync_instruments", lambda: ["A", "B"])
    monkeypatch.setattr(
        daily_cycle,
        "get_instruments",
        lambda *args, **kwargs: pd.DataFrame({"symbol": ["A", "B"], "instrument_key": ["K1", "K2"]}),
    )
    monkeypatch.setattr(daily_cycle, "capture_premarket", lambda: None)
    monkeypatch.setattr(
        daily_cycle,
        "ingest_symbols",
        lambda interval: [_ingestion_result("A", error="401 Unauthorized"), _ingestion_result("B", error="401 Unauthorized")],
    )
    monkeypatch.setattr(daily_cycle, "insert_research_run", lambda row: recorded_runs.append(row))

    with pytest.raises(IngestionFailure, match="2/2 symbols"):
        run_daily_cycle(mode="PAPER")

    assert len(recorded_runs) == 1
    assert recorded_runs[0]["status"] == "FAILED"
    assert "2/2 symbols" in recorded_runs[0]["error"]


def test_daily_cycle_tolerates_one_failed_symbol(monkeypatch):
    recorded_runs = []
    monkeypatch.setattr(daily_cycle, "sync_instruments", lambda: ["A", "B", "C"])
    monkeypatch.setattr(
        daily_cycle,
        "get_instruments",
        lambda *args, **kwargs: pd.DataFrame({"symbol": ["A", "B", "C"], "instrument_key": ["K1", "K2", "K3"]}),
    )
    monkeypatch.setattr(daily_cycle, "capture_premarket", lambda: None)
    monkeypatch.setattr(
        daily_cycle,
        "ingest_symbols",
        lambda interval: [_ingestion_result("A"), _ingestion_result("B"), _ingestion_result("C", error="timeout")],
    )
    monkeypatch.setattr(daily_cycle, "latest_market_context", lambda trading_date=None: {})
    # run_daily_cycle constructs UpstoxREST() inline as scan_top10's argument
    # *before* the (mocked) scan_top10 is ever called, so mocking scan_top10
    # alone doesn't stop the real client from being built -- and UpstoxREST()
    # raises without a real UPSTOX_ACCESS_TOKEN, which CI correctly doesn't
    # have. scan_top10 never inspects the client it's given here, so a plain
    # sentinel is enough to keep this test self-contained.
    monkeypatch.setattr(daily_cycle, "UpstoxREST", lambda: "FAKE_UPSTOX_CLIENT")
    monkeypatch.setattr(daily_cycle, "scan_top10", lambda api, config, trading_date: [])
    monkeypatch.setattr(daily_cycle, "insert_research_run", lambda row: recorded_runs.append(row))

    report = run_daily_cycle(mode="PAPER")

    assert report["candidate_count"] == 0
    assert len(recorded_runs) == 1
    assert recorded_runs[0]["status"] == "WARNING"
