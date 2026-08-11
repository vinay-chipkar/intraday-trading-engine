import pytest

from intraday_engine.signals.engine import SignalConfig, _structural_stop, generate_signal


def base_row():
    return {
        "close": 100.0,
        "atr14": 2.0,
        "ema9": 101.0,
        "ema20": 99.0,
        "ema50": 97.0,
        "vwap": 98.0,
        "rsi14": 60.0,
        "adx14": 30.0,
        "plus_di14": 28.0,
        "minus_di14": 15.0,
        "macd_histogram": 1.0,
        "relative_volume": 1.8,
        "distance_from_vwap_pct": 2.04,
        "trend": "UPTREND",
        "structure_trend": "UPTREND",
        "opening_range_breakout": True,
        "opening_range_breakdown": False,
        "hammer": True,
        "bullish_engulfing": False,
        "morning_star": False,
        "shooting_star": False,
        "bearish_engulfing": False,
        "evening_star": False,
        "double_bottom": False,
        "double_top": False,
    }


def test_bullish_setup_generates_buy_with_atr_risk_when_no_support_exists():
    signal = generate_signal(base_row(), market_score=8)
    assert signal.action == "BUY"
    assert signal.score >= 60
    assert signal.stop_loss == pytest.approx(98.0)
    assert signal.target == pytest.approx(103.0)
    assert signal.reward_risk == pytest.approx(1.5)


def test_structural_support_controls_long_stop_with_buffer():
    row = base_row()
    row["support"] = 96.0
    signal = generate_signal(row, market_score=8)

    assert signal.action == "BUY"
    assert signal.stop_loss == pytest.approx(95.8)
    assert signal.target == pytest.approx(106.3)
    assert signal.reward_risk == pytest.approx(1.5)
    assert "structural stop" in signal.reasons


def test_structural_stop_that_exceeds_risk_budget_blocks_trade():
    row = base_row()
    row["support"] = 90.0
    signal = generate_signal(row, market_score=8)

    assert signal.action == "NO_TRADE"
    assert "structural stop exceeds maximum risk" in signal.blockers


def test_bearish_setup_generates_sell():
    row = base_row()
    row.update({
        "ema9": 96.0,
        "ema20": 98.0,
        "ema50": 101.0,
        "vwap": 102.0,
        "rsi14": 40.0,
        "plus_di14": 12.0,
        "minus_di14": 28.0,
        "macd_histogram": -1.0,
        "trend": "DOWNTREND",
        "structure_trend": "DOWNTREND",
        "opening_range_breakout": False,
        "opening_range_breakdown": True,
        "hammer": False,
        "shooting_star": True,
    })
    signal = generate_signal(row, market_score=-8)
    assert signal.action == "SELL"
    assert signal.stop_loss == pytest.approx(102.0)
    assert signal.target == pytest.approx(97.0)


def test_structural_resistance_controls_short_stop_with_buffer():
    row = base_row()
    row.update({
        "ema9": 96.0,
        "ema20": 98.0,
        "ema50": 101.0,
        "vwap": 102.0,
        "rsi14": 40.0,
        "plus_di14": 12.0,
        "minus_di14": 28.0,
        "macd_histogram": -1.0,
        "trend": "DOWNTREND",
        "structure_trend": "DOWNTREND",
        "opening_range_breakout": False,
        "opening_range_breakdown": True,
        "hammer": False,
        "shooting_star": True,
        "resistance": 104.0,
    })
    signal = generate_signal(row, market_score=-8)
    assert signal.action == "SELL"
    assert signal.stop_loss == pytest.approx(104.2)
    assert signal.target == pytest.approx(93.7)


def test_weak_trend_blocks_trade_even_when_score_is_high():
    row = base_row()
    row["adx14"] = 10.0
    signal = generate_signal(row, market_score=10)
    assert signal.action == "NO_TRADE"
    assert "trend strength is too weak" in signal.blockers


def test_deterministic_output():
    row = base_row()
    first = generate_signal(row, market_score=5).as_dict()
    second = generate_signal(row, market_score=5).as_dict()
    assert first == second


def test_configurable_threshold_can_force_no_trade():
    signal = generate_signal(base_row(), market_score=0, config=SignalConfig(buy_threshold=99.0))
    assert signal.action == "NO_TRADE"


def test_structural_stop_rejects_wrong_side_stop_for_long():
    # Was the root cause behind the ~-258R/-49R/-30R degenerate trades found in
    # threshold-sweep diagnostics (BHARTIARTL/AXISBANK/ZOMATO): when the signal
    # bar's close has already broken below the support level computed from prior
    # bars (support=105 > entry=100), "support - buffer" can land above entry.
    # _structural_stop now rejects this the same way it already rejected
    # risk <= 0, instead of silently returning a stop on the wrong side of entry.
    row = {"support": 105.0}
    stop, reason = _structural_stop(row, side="LONG", entry=100.0, atr=5.0, config=SignalConfig())
    assert stop is None
    assert reason == "structural stop is on the wrong side of entry"


def test_structural_stop_rejects_wrong_side_stop_for_short():
    # Mirror image: entry has already broken above resistance computed from
    # prior bars, so "resistance + buffer" would land below entry for a SHORT.
    row = {"resistance": 95.0}
    stop, reason = _structural_stop(row, side="SHORT", entry=100.0, atr=5.0, config=SignalConfig())
    assert stop is None
    assert reason == "structural stop is on the wrong side of entry"


def test_structural_stop_valid_long_case_is_unchanged():
    row = {"support": 96.0}
    stop, reason = _structural_stop(row, side="LONG", entry=100.0, atr=2.0, config=SignalConfig())
    assert reason == "structural stop"
    assert stop == pytest.approx(95.8)


def test_structural_stop_valid_short_case_is_unchanged():
    row = {"resistance": 104.0}
    stop, reason = _structural_stop(row, side="SHORT", entry=100.0, atr=2.0, config=SignalConfig())
    assert reason == "structural stop"
    assert stop == pytest.approx(104.2)


def test_generate_signal_rejects_the_bhartiartl_style_wrong_side_stop_pattern():
    # End-to-end version of the two rejection tests above: reproduces the real
    # setup (entry already below its own recent support) through generate_signal
    # itself, proving such a signal can no longer reach backtest_signals with a
    # broken stop -- the degenerate R mechanism now has no way to occur via the
    # live/research signal-generation path.
    row = base_row()
    row["support"] = 105.0  # close (100.0) has already broken below this support
    signal = generate_signal(row, market_score=8)
    assert signal.action == "NO_TRADE"
    assert "structural stop is on the wrong side of entry" in signal.blockers
