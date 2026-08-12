from zoneinfo import ZoneInfo

import pytest

from intraday_engine.market.ingestion import IngestionFailure, IngestionResult
from scripts import paper_session


def _ingestion_result(symbol: str, *, error: str | None = None) -> IngestionResult:
    return IngestionResult(
        symbol=symbol,
        rows_received=0 if error else 5,
        rows_inserted=0 if error else 5,
        last_timestamp=None,
        quality={},
        error=error,
    )


def test_paper_session_rejects_invalid_interval(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["paper_session", "--interval", "0"],
    )
    with pytest.raises(ValueError, match="interval"):
        paper_session.main()


def test_paper_session_once_runs_bootstrap_refresh_observe_evaluate_and_summary(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(paper_session, "_bootstrap", lambda limit: calls.append(("bootstrap", limit)))
    monkeypatch.setattr(paper_session, "_refresh_candles", lambda: calls.append(("refresh",)))

    def fake_observe_once(*, limit, min_score):
        calls.append(("observe", limit, min_score))
        return [{"symbol": "TEST"}]

    def fake_evaluate_pending(*, max_holding_bars):
        calls.append(("evaluate", max_holding_bars))
        return {"pending": 1, "evaluated": 1, "waiting": 0}

    def fake_outcome_summary():
        calls.append(("summary",))
        return {"evaluated": 1, "wins": 1, "losses": 0, "win_rate": 1.0, "avg_r": 1.5}

    monkeypatch.setattr(paper_session, "observe_once", fake_observe_once)
    monkeypatch.setattr(paper_session, "evaluate_pending", fake_evaluate_pending)
    monkeypatch.setattr(paper_session, "outcome_summary", fake_outcome_summary)
    monkeypatch.setattr("sys.argv", ["paper_session", "--once", "--limit", "5", "--min-score", "55", "--max-holding-bars", "20"])

    paper_session.main()

    assert calls == [
        ("bootstrap", 5),
        ("refresh",),
        ("observe", 5, 55.0),
        ("evaluate", 20),
        ("summary",),
    ]
    assert "PAPER TICK observed=1 evaluated=1 waiting=0 total=1 wins=1 losses=0 avg_r=1.5000" in capsys.readouterr().out


def test_market_times_are_ist():
    assert paper_session._now().tzinfo == ZoneInfo("Asia/Kolkata")
    assert paper_session.MARKET_OPEN.hour == 9
    assert paper_session.MARKET_OPEN.minute == 15
    assert paper_session.MARKET_CLOSE.hour == 15
    assert paper_session.MARKET_CLOSE.minute == 30


def test_refresh_candles_raises_when_whole_universe_fails(monkeypatch):
    # Requirement: "if paper_session refresh fails for the whole universe, do
    # not silently continue" -- a few DATA WARNING prints used to be the only
    # signal; now the tick itself must fail.
    monkeypatch.setattr(
        paper_session,
        "ingest_symbols",
        lambda interval: [_ingestion_result(s, error="401 Unauthorized") for s in "ABCDE"],
    )
    with pytest.raises(IngestionFailure, match="5/5 symbols"):
        paper_session._refresh_candles()


def test_refresh_candles_tolerates_one_bad_symbol(monkeypatch):
    monkeypatch.setattr(
        paper_session,
        "ingest_symbols",
        lambda interval: [_ingestion_result(s) for s in "ABCD"] + [_ingestion_result("E", error="timeout")],
    )
    paper_session._refresh_candles()  # must not raise


def test_check_tick_freshness_raises_when_every_observation_is_stale():
    observed = [{"symbol": "A", "status": "STALE_DATA"}, {"symbol": "B", "status": "STALE_DATA"}]
    with pytest.raises(paper_session.StaleTickError, match="all 2 observations"):
        paper_session._check_tick_freshness(observed)


def test_check_tick_freshness_raises_on_zero_observations():
    with pytest.raises(paper_session.StaleTickError, match="zero observations"):
        paper_session._check_tick_freshness([])


def test_check_tick_freshness_tolerates_a_mix_of_stale_and_fresh():
    observed = [
        {"symbol": "A", "status": "STALE_DATA"},
        {"symbol": "B", "status": "NO_SIGNAL"},
        {"symbol": "C", "status": "SIGNAL_PENDING"},
    ]
    paper_session._check_tick_freshness(observed)  # must not raise


def test_fail_if_session_unhealthy_raises_when_every_tick_failed():
    with pytest.raises(SystemExit, match="UNHEALTHY"):
        paper_session._fail_if_session_unhealthy(ticks_ok=0, ticks_failed=3)


def test_fail_if_session_unhealthy_raises_when_majority_of_ticks_failed():
    with pytest.raises(SystemExit, match="UNHEALTHY"):
        paper_session._fail_if_session_unhealthy(ticks_ok=1, ticks_failed=2)


def test_fail_if_session_unhealthy_tolerates_a_healthy_majority():
    paper_session._fail_if_session_unhealthy(ticks_ok=8, ticks_failed=1)  # must not raise


def test_fail_if_session_unhealthy_noop_when_no_ticks_ran():
    paper_session._fail_if_session_unhealthy(ticks_ok=0, ticks_failed=0)  # must not raise
