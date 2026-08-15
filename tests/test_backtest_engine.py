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


def test_stop_wins_when_stop_and_target_are_both_inside_one_bar_short_side():
    # Long/short symmetry check for the same conservative assumption: a
    # SHORT's stop is above entry and target is below, so the mirrored bar
    # (low breaches target, high breaches stop) must still resolve as STOP.
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100, 101, 99, 100),
        ("2026-08-10 09:21:00", "TEST", 100, 103, 95, 97),
    ])
    result = backtest_signals([_signal(side="SHORT", stop=102, target=96)], bars)
    assert result.trades[0].outcome == "STOP"
    assert result.trades[0].exit_price == pytest.approx(102.0)
    assert result.trades[0].r_multiple == pytest.approx(-1.0)


def test_short_side_target_uses_next_bar_open_and_records_mfe_mae():
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100, 101, 99, 100),
        ("2026-08-10 09:21:00", "TEST", 99, 100, 96, 97),
    ])
    result = backtest_signals([_signal(side="SHORT", stop=102, target=97)], bars)
    trade = result.trades[0]
    assert trade.outcome == "TARGET"
    assert trade.entry_price == pytest.approx(99.0)
    assert trade.exit_price == pytest.approx(97.0)
    assert trade.r_multiple == pytest.approx(2.0 / 3.0)
    assert trade.mfe_points == pytest.approx(3.0)  # entry(99) - low(96)
    assert trade.mae_points == pytest.approx(1.0)  # high(100) - entry(99)


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


def test_wrong_side_structural_stop_can_produce_extreme_r_multiple_from_a_tiny_loss():
    # backtest_signals has no defense of its own against a wrong-side stop --
    # that fix now lives entirely at the signal-generation layer (see
    # test_structural_stop_rejects_wrong_side_stop_for_long/_short and
    # test_generate_signal_rejects_the_bhartiartl_style_wrong_side_stop_pattern in
    # test_signal_engine.py, which prove generate_signal can no longer produce
    # this signal in the first place). This test documents that residual,
    # intentionally-unaddressed property of backtest_signals itself: if *any*
    # caller hands it a signal whose stop is on the wrong side of its own entry,
    # nothing here validates that. `_simulate_one` fixes `risk` once, from
    # |next-bar fill price - stop|, and here the next bar's open happens to land
    # just 0.001 away from that stop, so a 0.20-point round-trip loss divides by
    # a ~0 risk denominator into a triple-digit R multiple.
    signal = _signal(stop=100.2, target=100.3, event_time="2026-08-10 09:20:00")
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100.0, 100.05, 99.95, 100.0),
        ("2026-08-10 09:21:00", "TEST", 100.099, 100.15, 99.9, 100.05),
    ])
    trade = backtest_signals([signal], bars, slippage_points=0.10).trades[0]
    assert trade.outcome == "STOP_GAP"
    assert trade.pnl_points == pytest.approx(-0.2)
    assert trade.r_multiple < -100  # a two-tick loss reported as a >100R blowup


def test_slippage_affects_pnl_but_not_outcome_classification():
    # Answers "does slippage affect outcome classification or only net P&L?":
    # outcome is decided purely from raw bar/stop/target price levels (see
    # `_simulate_one`'s open/high/low comparisons), independent of slippage_points.
    # Slippage is only applied afterwards, to the fill/exit price used for pnl.
    signal = _signal(stop=99.9, target=100.1, event_time="2026-08-10 09:20:00")
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100.0, 100.05, 99.95, 100.0),
        ("2026-08-10 09:21:00", "TEST", 100.05, 100.15, 99.98, 100.1),
    ])
    no_slip = backtest_signals([signal], bars, slippage_points=0.0).trades[0]
    with_slip = backtest_signals([signal], bars, slippage_points=0.10).trades[0]

    assert no_slip.outcome == with_slip.outcome == "TARGET"
    assert no_slip.pnl_points > 0
    assert with_slip.pnl_points < 0  # same price action, but the 0.20 round-trip slippage now exceeds the 0.10 raw entry-to-target margin


def test_target_hit_can_still_register_as_a_net_loss_when_reward_is_smaller_than_round_trip_slippage():
    # 43 TARGET + 14 TARGET_GAP trades in the real threshold-sweep diagnostics were
    # labeled as losses for exactly this reason. Nothing upstream (signal
    # generation or backtest execution) checks that a signal's reward exceeds
    # round-trip transaction cost before it is taken, so a trade whose target is
    # reached price-wise can still be a net loser once slippage is applied twice.
    signal = _signal(stop=99.9, target=100.1, event_time="2026-08-10 09:20:00")
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100.0, 100.05, 99.95, 100.0),
        ("2026-08-10 09:21:00", "TEST", 100.05, 100.15, 99.98, 100.1),
    ])
    trade = backtest_signals([signal], bars, slippage_points=0.10).trades[0]
    assert trade.outcome == "TARGET"
    assert trade.pnl_points < 0


def test_gap_through_target_on_the_entry_bar_is_a_deliberate_execution_model_choice():
    # This is a strategy-design decision, left unchanged: when the very bar used
    # for entry (next bar's open) has *already* gapped past target, the fill and
    # the gap-exit price are the same raw open, so the trade nets exactly
    # -2*slippage_points (a "wash") instead of capturing the gap.
    #
    # This was investigated as a candidate bug (falling through to the level-based
    # TARGET/STOP branches instead of exiting at the entry bar's own open) and
    # rejected: it makes things *worse*, not better. If the entry bar's open is
    # already beyond target, exiting at the stale target level instead of near the
    # open assumes a fill better than what the market actually offers (target is
    # further from the current price than the open is) -- e.g. entry gaps to
    # 100.45 with target=100.3, "fixing" this to exit at 100.3 would turn a bounded
    # -0.20 wash into a much larger loss for what was actually a favorable move.
    # Exiting near the open is the more realistic choice, so current behavior is
    # kept. Compare against a later-bar gap (test below), which correctly captures
    # the full gap value -- the difference is structural (entry and exit can't be
    # the same tick), not a defect in the gap-handling logic itself.
    signal = _signal(stop=99.8, target=100.1, event_time="2026-08-10 09:20:00")
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100.0, 100.05, 99.95, 100.0),
        ("2026-08-10 09:21:00", "TEST", 100.15, 100.2, 100.1, 100.18),
    ])
    trade = backtest_signals([signal], bars, slippage_points=0.10).trades[0]
    assert trade.outcome == "TARGET_GAP"
    assert trade.holding_bars == 1
    assert trade.pnl_points == pytest.approx(-0.2)  # exactly -2*slippage_points, zero captured movement


def test_gap_through_target_on_a_later_bar_captures_the_full_gap_value():
    # Contrast with the test above: the identical kind of gap (open >= target),
    # but occurring on a bar *after* the entry bar, has a genuinely separate
    # entry price and gap-exit price, so it correctly realizes the full gap.
    signal = _signal(stop=99.8, target=100.1, event_time="2026-08-10 09:20:00")
    bars = _bars([
        ("2026-08-10 09:20:00", "TEST", 100.0, 100.05, 99.95, 100.0),
        ("2026-08-10 09:21:00", "TEST", 100.0, 100.05, 99.95, 100.0),
        ("2026-08-10 09:22:00", "TEST", 100.6, 100.7, 100.5, 100.65),
    ])
    trade = backtest_signals([signal], bars, slippage_points=0.10).trades[0]
    assert trade.outcome == "TARGET_GAP"
    assert trade.holding_bars == 2
    assert trade.pnl_points > 0.3  # captures most of the 0.6-point gap, not a wash
