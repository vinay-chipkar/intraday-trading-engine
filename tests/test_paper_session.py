from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts import paper_session


def test_paper_session_rejects_invalid_interval(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["paper_session", "--interval", "0"],
    )
    with pytest.raises(ValueError, match="interval"):
        paper_session.main()


def test_paper_session_once_runs_observe_evaluate_and_summary(monkeypatch, capsys):
    calls = []

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

    assert calls == [("observe", 5, 55.0), ("evaluate", 20), ("summary",)]
    assert "PAPER TICK observed=1 evaluated=1 waiting=0 total=1 wins=1 losses=0 avg_r=1.5000" in capsys.readouterr().out


def test_market_times_are_ist():
    assert paper_session._now().tzinfo == ZoneInfo("Asia/Kolkata")
    assert paper_session.MARKET_OPEN.hour == 9
    assert paper_session.MARKET_OPEN.minute == 15
    assert paper_session.MARKET_CLOSE.hour == 15
    assert paper_session.MARKET_CLOSE.minute == 30
