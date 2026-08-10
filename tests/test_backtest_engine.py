import pandas as pd
import pytest

from intraday_engine.backtest import backtest_signals
from intraday_engine.signals.engine import TradeSignal


def _signal(side="LONG", event_time="2026-08-10 09:20:00", stop=98.0, target=104.0):
    return TradeSignal(
        action="BUY" if side == "LONG" else "SELL",
        score=80.0 if side == "LONG" else -80.0,
        confidence=80.0,
        entry=100.0,
        stop_loss=stop,
        target=target,
        reward_risk=abs(target - 100.0) / abs(100.0 - stop),
        reasons=("TREND",),
        blockers=(),
        symbol="TEST",
        event_time=pd.Timestamp(event_time),
    )


def _bars(rows):
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close"])


def test_signal_executes_on_next_bar_not_signal_bar():
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100, 110, 99, 109),
        ("2026-08-10 09:21:00", "TEST", 101, 103, 100, 102),
        ("2026-08-10 09:22:00", "TEST", 102, 103, 101, 102),
    ])
    result = backtest_signals([_signal(target=104)], bars)
    assert result.total_trades == 1
    assert result.trades[0].entry_time == pd.Timestamp("2026-08-10 09:21:00")
    assert result.trades[0].outcome == "TIMEOUT"


def test_stop_wins_when_stop_and_target_are_both_inside_one_bar():
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100, 101, 99, 100),
        ("2026-08-10 09:21:00", "TEST", 100, 105, 97, 103),
    ])
    result = backtest_signals([_signal(stop=98, target=104)], bars)
    assert result.trades[0].outcome == "STOP"
    assert result.trades[0].exit_price == pytest.approx(98.0)
    assert result.trades[0].r_multiple == pytest.approx(-1.0)


def test_target_and_metrics_are_deterministic():
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100, 101, 99, 100),
        ("2026-08-10 09:21:00", "TEST", 101, 105, 100, 104),
    ])
    result = backtest_signals([_signal(stop=98, target=104)], bars)
    assert result.wins == 1
    assert result.losses == 0
    assert result.win_rate == pytest.approx(1.0)
    assert result.profit_factor == float("inf")
    assert result.net_points == pytest.approx(3.0)
    assert result.expectancy_r == pytest.approx(1.0)
