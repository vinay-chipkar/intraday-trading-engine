import pytest

from intraday_engine.backtest.costs import (
    CostModel,
    apply_cost_model,
    backtest_trades_to_cost_rows,
    build_cost_comparison_report,
)


def _trade(side="LONG", entry=100.0, exit=103.0, stop=98.0, pnl=None):
    pnl = pnl if pnl is not None else (exit - entry if side == "LONG" else entry - exit)
    return {"side": side, "entry_price": entry, "exit_price": exit, "stop_loss": stop, "pnl_points": pnl}


def test_zero_cost_preset_leaves_net_equal_to_gross():
    report = apply_cost_model([_trade()], CostModel.zero_cost())
    assert report.slippage_points == 0.0
    assert report.transaction_cost_points == 0.0
    assert report.net_pnl_points == pytest.approx(report.gross_pnl_points)
    assert report.net_expectancy_r == pytest.approx(report.gross_expectancy_r)


def test_realistic_preset_reduces_net_pnl_below_gross_for_a_profitable_trade():
    report = apply_cost_model([_trade(entry=100.0, exit=103.0)], CostModel.realistic())
    assert report.gross_pnl_points == pytest.approx(3.0)
    assert report.transaction_cost_points > 0.0
    assert report.net_pnl_points < report.gross_pnl_points


def test_conservative_stress_costs_more_than_realistic():
    realistic = apply_cost_model([_trade()], CostModel.realistic())
    stressed = apply_cost_model([_trade()], CostModel.conservative_stress())
    assert stressed.transaction_cost_points > realistic.transaction_cost_points
    assert stressed.slippage_points > realistic.slippage_points
    assert stressed.net_pnl_points < realistic.net_pnl_points


def test_stt_applies_only_to_the_sell_leg_long_vs_short():
    model = CostModel.realistic()
    # LONG: entry=BUY (no STT), exit=SELL (STT applies)
    long_cost = model.round_trip_cost_points(100.0, 103.0, "LONG")
    # SHORT: entry=SELL (STT applies), exit=BUY (no STT) -- same prices, same STT total
    short_cost = model.round_trip_cost_points(103.0, 100.0, "SHORT")
    assert long_cost == pytest.approx(short_cost)

    # Isolate STT specifically: a model with only stt_sell_pct set
    stt_only = CostModel(brokerage_pct=0, exchange_txn_pct=0, sebi_pct=0, gst_pct=0, stamp_duty_buy_pct=0, stt_sell_pct=0.001)
    assert stt_only.leg_cost_points(100.0, is_buy_leg=True) == pytest.approx(0.0)
    assert stt_only.leg_cost_points(100.0, is_buy_leg=False) == pytest.approx(0.1)


def test_stamp_duty_applies_only_to_the_buy_leg():
    stamp_only = CostModel(brokerage_pct=0, exchange_txn_pct=0, sebi_pct=0, gst_pct=0, stt_sell_pct=0, stamp_duty_buy_pct=0.001)
    assert stamp_only.leg_cost_points(100.0, is_buy_leg=False) == pytest.approx(0.0)
    assert stamp_only.leg_cost_points(100.0, is_buy_leg=True) == pytest.approx(0.1)


def test_gst_applies_only_to_brokerage_and_exchange_charges():
    model = CostModel(brokerage_pct=0.001, exchange_txn_pct=0.0005, sebi_pct=0, stt_sell_pct=0, stamp_duty_buy_pct=0, gst_pct=0.18)
    # brokerage(0.001) + exchange(0.0005) = 0.0015; GST adds 18% of that = 0.00027; total = 0.00177
    expected_pct = 0.001 + 0.0005 + (0.001 + 0.0005) * 0.18
    assert model.leg_cost_points(100.0, is_buy_leg=True) == pytest.approx(100.0 * expected_pct)


def test_round_trip_slippage_is_entry_plus_exit():
    model = CostModel(entry_slippage_points=0.1, exit_slippage_points=0.15)
    assert model.round_trip_slippage_points() == pytest.approx(0.25)


def test_brokerage_is_an_uncapped_linear_percentage_not_a_per_order_cap():
    # Documents/locks in the fix: brokerage_pct is a flat percentage of
    # turnover with no ceiling, because this module has no order-value/
    # quantity information to evaluate a real "percentage OR flat fee,
    # whichever is lower" cap against (see CostModel's docstring). Brokerage
    # cost must scale exactly linearly with price, with no cap kicking in
    # even at a price where a real per-order cap would clearly have bound.
    brokerage_only = CostModel(
        brokerage_pct=0.0003, exchange_txn_pct=0, sebi_pct=0, stt_sell_pct=0, stamp_duty_buy_pct=0, gst_pct=0
    )
    low_price_cost = brokerage_only.leg_cost_points(100.0, is_buy_leg=True)
    high_price_cost = brokerage_only.leg_cost_points(1_000_000.0, is_buy_leg=True)

    assert low_price_cost == pytest.approx(100.0 * 0.0003)
    # A real discount-broker cap (e.g. Rs 20 flat OR 0.03% of turnover,
    # whichever is lower) would have capped the high-price leg at a small
    # flat fee; this implementation instead scales it up by the same factor
    # as the price increased, proving no cap is silently applied.
    assert high_price_cost == pytest.approx(1_000_000.0 * 0.0003)
    assert high_price_cost / low_price_cost == pytest.approx(1_000_000.0 / 100.0)


def test_invalid_side_raises():
    model = CostModel.realistic()
    with pytest.raises(ValueError, match="LONG or SHORT"):
        model.round_trip_cost_points(100.0, 103.0, "SIDEWAYS")


def test_empty_trades_returns_zeroed_report():
    report = apply_cost_model([], CostModel.realistic())
    assert report.trades == 0
    assert report.gross_pnl_points == 0.0
    assert report.gross_profit_factor == 0.0


def test_apply_cost_model_computes_profit_factor_and_expectancy_across_trades():
    trades = [
        _trade(entry=100.0, exit=103.0, stop=98.0),   # win, risk=2, gross_r=1.5
        _trade(entry=100.0, exit=98.0, stop=98.0, pnl=-2.0),  # loss, risk=2, gross_r=-1.0
    ]
    report = apply_cost_model(trades, CostModel.zero_cost())
    assert report.trades == 2
    assert report.gross_expectancy_r == pytest.approx((1.5 + -1.0) / 2)
    assert report.gross_profit_factor == pytest.approx(3.0 / 2.0)


def test_backtest_trades_to_cost_rows_adapts_real_backtest_trades():
    from intraday_engine.backtest.engine import backtest_signals
    from intraday_engine.signals.engine import TradeSignal
    import pandas as pd

    signal = TradeSignal(
        action="BUY", score=80.0, confidence=80.0, entry=100.0,
        stop_loss=98.0, target=104.0, reward_risk=2.0,
        reasons=(), blockers=(), symbol="TEST", event_time=pd.Timestamp("2026-08-10 09:20:00"),
    )
    bars = pd.DataFrame([
        ("2026-08-10 09:20:00", "TEST", 100, 101, 99, 100),
        ("2026-08-10 09:21:00", "TEST", 101, 105, 100, 104),
    ], columns=["timestamp", "symbol", "open", "high", "low", "close"])

    result = backtest_signals([signal], bars)
    rows = backtest_trades_to_cost_rows(result.trades)
    report = apply_cost_model(rows, CostModel.realistic())
    assert report.trades == 1
    assert report.gross_pnl_points == pytest.approx(result.trades[0].pnl_points)


def test_cost_comparison_report_produces_all_nine_metrics_consistently():
    trades = [
        _trade(entry=100.0, exit=103.0, stop=98.0),
        _trade(entry=100.0, exit=98.0, stop=98.0, pnl=-2.0),
    ]
    comparison = build_cost_comparison_report(trades)

    # Cross-check against apply_cost_model directly for both presets.
    realistic = apply_cost_model(trades, CostModel.realistic())
    conservative = apply_cost_model(trades, CostModel.conservative_stress())

    assert comparison.trades == 2
    assert comparison.gross_pnl_points == pytest.approx(realistic.gross_pnl_points)
    assert comparison.gross_pnl_points == pytest.approx(conservative.gross_pnl_points)  # gross is preset-independent
    assert comparison.gross_expectancy_r == pytest.approx(realistic.gross_expectancy_r)
    assert comparison.gross_profit_factor == pytest.approx(realistic.gross_profit_factor)

    assert comparison.realistic_net_pnl_points == pytest.approx(realistic.net_pnl_points)
    assert comparison.realistic_net_expectancy_r == pytest.approx(realistic.net_expectancy_r)
    assert comparison.realistic_net_profit_factor == pytest.approx(realistic.net_profit_factor)

    assert comparison.conservative_net_pnl_points == pytest.approx(conservative.net_pnl_points)
    assert comparison.conservative_net_expectancy_r == pytest.approx(conservative.net_expectancy_r)
    assert comparison.conservative_net_profit_factor == pytest.approx(conservative.net_profit_factor)

    # Costs only ever erode P&L here, never improve it.
    assert comparison.conservative_net_pnl_points < comparison.realistic_net_pnl_points < comparison.gross_pnl_points


def test_cost_comparison_report_handles_empty_trades():
    comparison = build_cost_comparison_report([])
    assert comparison.trades == 0
    assert comparison.gross_pnl_points == 0.0
    assert comparison.realistic_net_pnl_points == 0.0
    assert comparison.conservative_net_pnl_points == 0.0
    assert comparison.gross_profit_factor == 0.0
