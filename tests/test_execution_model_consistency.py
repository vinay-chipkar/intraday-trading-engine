"""backtest/engine.py and research/paper_outcomes.py are two independent
trade simulators (see CLAUDE.md) that are each expected to report
mae_points/mfe_points on the same scale: non-negative excursion magnitudes.
Before EXECUTION_MODEL_VERSION "1.1.0", paper_outcomes.py's mae_points was
signed (negative for an adverse move) while backtest/engine.py's was always
a non-negative magnitude -- the two silently disagreed on what "mae_points"
even meant. These tests pin the same LONG/SHORT scenario through both
simulators and assert the reported mae_points/mfe_points match exactly."""

from __future__ import annotations

import pandas as pd
import pytest

from intraday_engine.backtest import backtest_signals
from intraday_engine.research.paper_outcomes import evaluate_trade
from intraday_engine.signals.engine import TradeSignal
from intraday_engine.versioning import EXECUTION_MODEL_VERSION


def _signal(side="LONG", stop=98.0, target=104.0):
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
        event_time=pd.Timestamp("2026-08-10 09:20:00"),
    )


def _observation(action="BUY", stop=98.0, target=104.0):
    return {
        "observation_id": "obs-test",
        "bar_time": pd.Timestamp("2026-08-10 09:20:00"),
        "signal_action": action,
        "entry_price": 100.0,
        "stop_loss": stop,
        "target": target,
    }


def _backtest_bars(rows):
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close"])


def _paper_bars(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_mae_and_mfe_agree_between_backtest_and_paper_outcomes(side):
    stop, target = (90.0, 110.0) if side == "LONG" else (110.0, 90.0)
    # Timeout scenario (never touches stop/target) so both simulators walk
    # the full window and accumulate mfe/mae the same way.
    timeout_rows = [
        ("2026-08-10 09:20:00", 100, 101, 99, 100),
        ("2026-08-10 09:21:00", 100, 103, 97, 101),
    ]
    backtest_result = backtest_signals(
        [_signal(side=side, stop=stop, target=target)],
        _backtest_bars([(ts, "TEST", o, h, low, c) for ts, o, h, low, c in timeout_rows]),
        max_holding_bars=1,
    )
    paper_result = evaluate_trade(
        _observation(action="BUY" if side == "LONG" else "SELL", stop=stop, target=target),
        _paper_bars([(ts, o, h, low, c, 1000) for ts, o, h, low, c in timeout_rows]),
        max_holding_bars=1,
    )

    assert backtest_result.trades[0].outcome == "TIMEOUT"
    assert paper_result["outcome"] == "TIMEOUT"
    assert paper_result["mae_points"] == pytest.approx(backtest_result.trades[0].mae_points)
    assert paper_result["mfe_points"] == pytest.approx(backtest_result.trades[0].mfe_points)
    assert paper_result["mae_points"] >= 0.0
    assert paper_result["mfe_points"] >= 0.0


def test_paper_outcomes_rows_are_stamped_with_the_mae_sign_convention_version():
    result = evaluate_trade(
        _observation(),
        _paper_bars([
            ("2026-08-10 09:20:00", 100, 101, 99, 100, 1000),
            ("2026-08-10 09:21:00", 101, 104, 100, 103, 1000),
        ]),
    )
    # Provenance, not a rewrite: this is how a future reader distinguishes
    # this row's (post-fix) non-negative mae_points from a pre-fix row's
    # signed mae_points without having to guess from the value alone.
    assert result["execution_model_version"] == EXECUTION_MODEL_VERSION == "1.1.0"
