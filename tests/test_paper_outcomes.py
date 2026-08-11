import pandas as pd
import pytest

from intraday_engine.research.paper_outcomes import evaluate_trade


def _observation(action="BUY"):
    return {
        "observation_id": "obs-test",
        "bar_time": pd.Timestamp("2026-08-11 09:20:00+05:30"),
        "signal_action": action,
        "entry_price": 100.0,
        "stop_loss": 98.0 if action == "BUY" else 102.0,
        "target": 103.0 if action == "BUY" else 97.0,
    }


def _bars(rows):
    return pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def test_long_target_uses_next_bar_open_and_records_mfe_mae():
    bars = _bars([
        ("2026-08-11 09:20:00+05:30", 100, 101, 99, 100, 1000),
        ("2026-08-11 09:21:00+05:30", 101, 104, 100, 103, 1000),
    ])
    result = evaluate_trade(_observation(), bars)
    assert result["outcome"] == "TARGET"
    assert result["entry_price"] == pytest.approx(101.0)
    assert result["exit_price"] == pytest.approx(103.0)
    assert result["r_multiple"] == pytest.approx(2.0 / 3.0)
    assert result["mfe_points"] == pytest.approx(3.0)
    assert result["mae_points"] == pytest.approx(-1.0)


def test_stop_wins_when_stop_and_target_are_both_touched():
    bars = _bars([
        ("2026-08-11 09:20:00+05:30", 100, 101, 99, 100, 1000),
        ("2026-08-11 09:21:00+05:30", 100, 104, 97, 101, 1000),
    ])
    result = evaluate_trade(_observation(), bars)
    assert result["outcome"] == "STOP"
    assert result["exit_price"] == pytest.approx(98.0)
    assert result["r_multiple"] == pytest.approx(-1.0)


def test_waits_until_full_holding_window_exists():
    bars = _bars([
        ("2026-08-11 09:20:00+05:30", 100, 101, 99, 100, 1000),
        ("2026-08-11 09:21:00+05:30", 101, 102, 100, 101, 1000),
    ])
    assert evaluate_trade(_observation(), bars, max_holding_bars=2) is None
