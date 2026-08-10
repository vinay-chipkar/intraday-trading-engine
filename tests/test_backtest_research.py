import pandas as pd

from intraday_engine.backtest.research import (
    FilterSpec,
    _adx,
    _aligned_emas,
    _macd,
    _trend,
    _vwap,
    filter_signals,
)
from intraday_engine.signals.engine import TradeSignal


def _row(**values):
    base = {
        "close": 101.0,
        "vwap": 100.0,
        "ema9": 100.5,
        "ema20": 100.0,
        "ema50": 99.0,
        "trend": "UPTREND",
        "adx14": 30.0,
        "plus_di14": 25.0,
        "minus_di14": 15.0,
        "macd_histogram": 0.5,
    }
    base.update(values)
    return pd.Series(base)


def test_long_feature_predicates_accept_bullish_alignment():
    row = _row()
    assert _trend(row, "LONG")
    assert _vwap(row, "LONG")
    assert _aligned_emas(row, "LONG")
    assert _adx(row, "LONG")
    assert _macd(row, "LONG")


def test_short_feature_predicates_reject_bullish_alignment():
    row = _row()
    assert not _trend(row, "SHORT")
    assert not _vwap(row, "SHORT")
    assert not _aligned_emas(row, "SHORT")
    assert not _adx(row, "SHORT")
    assert not _macd(row, "SHORT")


def test_filter_signals_uses_features_at_signal_time_only():
    first = pd.Timestamp("2026-08-10 09:20:00")
    second = pd.Timestamp("2026-08-10 09:21:00")
    signal = TradeSignal(
        action="BUY",
        score=70.0,
        confidence=70.0,
        entry=100.0,
        stop_loss=98.0,
        target=103.0,
        reward_risk=1.5,
        reasons=(),
        blockers=(),
        symbol="TEST",
        event_time=first,
    )
    features = pd.DataFrame(
        [
            {"timestamp": first, "trend": "UPTREND"},
            {"timestamp": second, "trend": "DOWNTREND"},
        ]
    )
    spec = FilterSpec("uptrend", lambda row, side: row["trend"] == "UPTREND")

    assert filter_signals([signal], features, spec.predicate) == [signal]
    assert filter_signals([signal], features, spec.predicate, start_time=second) == []
