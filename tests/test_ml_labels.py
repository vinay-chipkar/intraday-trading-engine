import pandas as pd
import pytest

from intraday_engine.ml.labels import build_forward_labels, label_forward_path


def bars(rows):
    return pd.DataFrame(rows, columns=["timestamp", "high", "low"])


def test_long_target_before_stop():
    df = bars([("2026-01-01T09:16:00Z", 101, 99), ("2026-01-01T09:17:00Z", 103, 100)])
    result = label_forward_path(df, 100, 98, 102, side="LONG")
    assert result["label"] == 1
    assert result["exit_reason"] == "TARGET"
    assert result["bars_to_exit"] == 2


def test_short_target_before_stop():
    df = bars([("2026-01-01T09:16:00Z", 101, 99), ("2026-01-01T09:17:00Z", 98, 96)])
    result = label_forward_path(df, 100, 102, 97, side="SHORT")
    assert result["label"] == 1
    assert result["exit_reason"] == "TARGET"


def test_same_bar_stop_and_target_is_conservative_stop():
    df = bars([("2026-01-01T09:16:00Z", 103, 97)])
    result = label_forward_path(df, 100, 98, 102, side="LONG")
    assert result["label"] == 0
    assert result["exit_reason"] == "STOP"


def test_builder_excludes_entry_candle_from_future_path():
    df = bars([
        ("2026-01-01T09:15:00Z", 110, 90),
        ("2026-01-01T09:16:00Z", 101, 99),
        ("2026-01-01T09:17:00Z", 103, 100),
    ])
    entries = pd.DataFrame([{"timestamp": "2026-01-01T09:15:00Z", "entry": 100, "stop": 98, "target": 102, "side": "LONG"}])
    result = build_forward_labels(df, entries, horizon=2)
    assert result.iloc[0]["label"] == 1
    assert result.iloc[0]["bars_to_exit"] == 2


def test_invalid_side_rejected():
    df = bars([("2026-01-01T09:16:00Z", 101, 99)])
    with pytest.raises(ValueError):
        label_forward_path(df, 100, 98, 102, side="BUY")
