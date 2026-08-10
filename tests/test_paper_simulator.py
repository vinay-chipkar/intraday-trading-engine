from datetime import datetime

import pytest

from intraday_engine.paper.simulator import PaperBroker


DAY = datetime(2026, 8, 10, 9, 20)
NEXT_DAY = datetime(2026, 8, 11, 9, 20)


def test_paper_broker_sizes_by_risk_and_notional_cap():
    broker = PaperBroker(
        100_000,
        risk_per_trade=0.01,
        max_position_notional=0.25,
    )
    position = broker.open("TEST", "LONG", 100.0, 98.0, 104.0, now=DAY)

    assert position is not None
    assert position.quantity == 250
    assert position.entry == pytest.approx(100.0)


def test_paper_broker_enforces_max_positions():
    broker = PaperBroker(100_000, max_positions=1)
    first = broker.open("A", "LONG", 100.0, 99.0, 102.0, now=DAY)
    second = broker.open("B", "LONG", 100.0, 99.0, 102.0, now=DAY)

    assert first is not None
    assert second is None


def test_daily_loss_halts_new_positions():
    broker = PaperBroker(
        100_000,
        risk_per_trade=0.01,
        max_daily_loss=0.01,
        max_position_notional=0.25,
    )
    position = broker.open("TEST", "LONG", 100.0, 96.0, 108.0, now=DAY)
    assert position is not None

    result = broker.mark("TEST", 96.0, now=DAY)
    assert result is not None
    assert result["reason"] == "STOP"
    assert broker.halted
    assert broker.realized_pnl_today < 0

    blocked = broker.open("NEXT", "LONG", 100.0, 99.0, 102.0, now=DAY)
    assert blocked is None


def test_daily_guard_resets_on_next_trading_day():
    broker = PaperBroker(
        100_000,
        risk_per_trade=0.01,
        max_daily_loss=0.01,
        max_position_notional=0.25,
    )
    broker.open("TEST", "LONG", 100.0, 96.0, 108.0, now=DAY)
    broker.mark("TEST", 96.0, now=DAY)
    assert broker.halted

    status = broker.status(now=NEXT_DAY)
    assert not status["halted"]
    assert status["realized_pnl_today"] == 0.0
