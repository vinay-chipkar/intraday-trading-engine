import pandas as pd

from intraday_engine.research.paper_observer import build_observation
from intraday_engine.signals.engine import TradeSignal


def _candidate():
    return {
        "symbol": "TEST",
        "instrument_key": "NSE_EQ|TEST",
        "rank": 1,
        "candidate_score": 42.5,
        "change_pct": -0.5,
        "relative_volume": 1.2,
        "vwap": 100.5,
    }


def _signal(action="NO_TRADE"):
    return TradeSignal(
        action=action,
        score=-42.0 if action == "NO_TRADE" else -72.0,
        confidence=42.0 if action == "NO_TRADE" else 72.0,
        entry=100.0,
        stop_loss=101.0 if action == "SELL" else None,
        target=98.5 if action == "SELL" else None,
        reward_risk=1.5 if action == "SELL" else None,
        reasons=("EMA trend is bearish",),
        blockers=() if action == "SELL" else ("trend strength is too weak",),
        symbol="TEST",
        event_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )


def test_no_signal_observation_is_safe_and_has_no_entry():
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    assert row["signal_action"] == "NO_TRADE"
    assert row["status"] == "NO_SIGNAL"
    assert row["entry_price"] is None
    assert "trend strength is too weak" in row["signal_blockers"]


def test_signal_observation_records_trade_plan():
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal("SELL"),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    assert row["signal_action"] == "SELL"
    assert row["status"] == "SIGNAL_PENDING"
    assert row["entry_price"] == 100.0
    assert row["stop_loss"] == 101.0
    assert row["target"] == 98.5
