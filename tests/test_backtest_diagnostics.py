import pandas as pd

from intraday_engine.backtest.diagnostics import summarize_diagnostics, trade_diagnostics
from intraday_engine.backtest.engine import BacktestResult, BacktestTrade
from intraday_engine.signals.engine import TradeSignal


def test_trade_diagnostics_preserves_signal_metadata():
    signal = TradeSignal(
        action="BUY",
        score=72.0,
        confidence=72.0,
        entry=100.0,
        stop_loss=98.0,
        target=103.0,
        reward_risk=1.5,
        reasons=("EMA trend is bullish", "price is above VWAP"),
        blockers=(),
        symbol="TEST",
        event_time=pd.Timestamp("2026-08-10 09:20:00"),
    )
    trade = BacktestTrade(
        symbol="TEST",
        side="LONG",
        signal_time=signal.event_time,
        entry_time=pd.Timestamp("2026-08-10 09:21:00"),
        exit_time=pd.Timestamp("2026-08-10 09:24:00"),
        entry_price=100.0,
        exit_price=103.0,
        stop_loss=98.0,
        target=103.0,
        outcome="TARGET",
        pnl_points=3.0,
        r_multiple=1.5,
        holding_bars=4,
    )
    result = BacktestResult((trade,), 1, 1, 0, 0, 1.0, float("inf"), 3.0, 0.0, 1.5)

    diagnostics = trade_diagnostics(result, [signal])

    assert diagnostics.loc[0, "signal_score"] == 72.0
    assert diagnostics.loc[0, "confidence"] == 72.0
    assert "price is above VWAP" in diagnostics.loc[0, "reasons"]


def test_summarize_diagnostics_groups_side_and_score():
    trades = pd.DataFrame(
        [
            {"symbol": "TEST", "side": "LONG", "pnl_points": 3.0, "r_multiple": 1.5, "signal_score": 72.0, "reasons": "EMA trend is bullish | price is above VWAP"},
            {"symbol": "TEST", "side": "LONG", "pnl_points": -2.0, "r_multiple": -1.0, "signal_score": 65.0, "reasons": "EMA trend is bullish"},
        ]
    )

    by_side, by_score, by_reason = summarize_diagnostics(trades)

    assert by_side.loc[0, "trades"] == 2
    assert by_side.loc[0, "net_points"] == 1.0
    assert by_score["trades"].sum() == 2
    assert not by_reason.empty
