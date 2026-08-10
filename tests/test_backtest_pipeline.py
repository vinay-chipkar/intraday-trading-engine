import pandas as pd

from intraday_engine.backtest.pipeline import run_rule_backtest


def test_run_rule_backtest_adds_symbol_before_execution():
    bars = pd.DataFrame(
        [
            ("2026-08-10 09:20:00", 100, 101, 99, 100, 1000),
            ("2026-08-10 09:21:00", 100, 105, 99, 104, 1000),
            ("2026-08-10 09:22:00", 104, 105, 103, 104, 1000),
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    result = run_rule_backtest(
        bars,
        symbol="TEST",
        min_score=60.0,
    )

    assert all(trade.symbol == "TEST" for trade in result.trades)
