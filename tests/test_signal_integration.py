import pandas as pd

from intraday_engine.signals.engine import TradeSignal
from intraday_engine.strategy.point_in_time import generate_signals


def _bars() -> pd.DataFrame:
    rows = []
    for i in range(80):
        close = 100.0 + i * 0.25
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-08-10 09:15") + pd.Timedelta(minutes=i),
                "symbol": "TEST",
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 100_000,
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_returns_only_canonical_trade_signals():
    signals = generate_signals(_bars(), symbol="TEST", market_score=10, min_score=60, pivot_left=2, pivot_right=2)
    assert all(isinstance(signal, TradeSignal) for signal in signals)
    assert all(signal.symbol == "TEST" for signal in signals)
    assert all(signal.action in {"BUY", "SELL"} for signal in signals)
    assert all(signal.event_time is not None for signal in signals)
