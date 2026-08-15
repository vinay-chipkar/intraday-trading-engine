import dataclasses

import pandas as pd
import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.research.paper_observer import _latest_context, build_observation, is_bar_stale
from intraday_engine.signals.engine import TradeSignal


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "test.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)
    return fake_settings.duckdb_path


def _candidate():
    return {
        "symbol": "TEST",
        "instrument_key": "NSE_EQ|TEST",
        "rank": 1,
        "candidate_score": 42.5,
        "change_pct": -0.5,
        "relative_volume": 1.2,
        "vwap": 100.5,
    }


def _signal(action="NO_TRADE"):
    return TradeSignal(
        action=action,
        score=-42.0 if action == "NO_TRADE" else -72.0,
        confidence=42.0 if action == "NO_TRADE" else 72.0,
        entry=100.0,
        stop_loss=101.0 if action == "SELL" else None,
        target=98.5 if action == "SELL" else None,
        reward_risk=1.5 if action == "SELL" else None,
        reasons=("EMA trend is bearish",),
        blockers=() if action == "SELL" else ("trend strength is too weak",),
        symbol="TEST",
        event_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )


def test_new_observations_are_stamped_with_provenance_versions():
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    from intraday_engine.versioning import FEATURE_ENGINE_VERSION, STRATEGY_VERSION

    assert row["strategy_version"] == STRATEGY_VERSION
    assert row["feature_engine_version"] == FEATURE_ENGINE_VERSION
    # code_commit is best-effort (None outside a git checkout) -- just must
    # not raise and must be present as a key.
    assert "code_commit" in row


def test_no_signal_observation_is_safe_and_has_no_entry():
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    assert row["signal_action"] == "NO_TRADE"
    assert row["status"] == "NO_SIGNAL"
    assert row["entry_price"] is None
    assert "trend strength is too weak" in row["signal_blockers"]


def test_signal_observation_records_trade_plan():
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal("SELL"),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    assert row["signal_action"] == "SELL"
    assert row["status"] == "SIGNAL_PENDING"
    assert row["entry_price"] == 100.0
    assert row["stop_loss"] == 101.0
    assert row["target"] == 98.5


def test_is_bar_stale_detects_a_prior_trading_day_bar():
    # Reproduces the real 2026-08-12 incident: intraday ingestion for today
    # never ran, so the last available 1m bar for HINDALCO/SBIN/TCS was from
    # 2026-08-11, while the scanner candidate was already scored from live,
    # same-day quotes.
    assert is_bar_stale(pd.Timestamp("2026-08-11 05:56:00+00:00"), trading_date=pd.Timestamp("2026-08-12").date())


def test_is_bar_stale_false_for_a_same_day_bar():
    assert not is_bar_stale(pd.Timestamp("2026-08-12 04:10:00+00:00"), trading_date=pd.Timestamp("2026-08-12").date())


def test_stale_bar_forces_stale_status_and_strips_any_trade_plan():
    # Even if generate_signal happened to return an actionable BUY/SELL from a
    # stale (wrong trading day) bar, it must never be surfaced as a real trade
    # plan -- the signal is about a day that has already closed, not "now".
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-12 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-12").date(),
        candidate=_candidate(),
        signal=_signal("SELL"),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
        stale=True,
    )
    assert row["status"] == "STALE_DATA"
    assert row["entry_price"] is None
    assert row["stop_loss"] is None
    assert row["target"] is None
    # the raw signal action/score are preserved for diagnostics, just not acted on
    assert row["signal_action"] == "SELL"


def _insert_market_context(db_path: str, *, trading_date: str, captured_at: str, regime: str, score: float) -> None:
    connection = db.conn(path=db_path)
    try:
        connection.execute(
            f"""
            INSERT INTO market_context VALUES
                (TIMESTAMPTZ '{captured_at}', DATE '{trading_date}',
                 NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                 0, 0, ?, ?)
            """,
            [score, regime],
        )
    finally:
        connection.close()


def test_genuine_neutral_context_is_distinguishable_from_missing_context(isolated_db):
    # A) genuine NEUTRAL market: a real premarket capture exists for today,
    # with regime literally set to the string "NEUTRAL".
    _insert_market_context(
        isolated_db, trading_date="2026-08-11", captured_at="2026-08-11 03:45:00+00", regime="NEUTRAL", score=0.0
    )
    available = _latest_context(pd.Timestamp("2026-08-11").date())
    assert available == {"regime": "NEUTRAL", "score": 0.0, "status": "AVAILABLE"}

    # B) missing context: no premarket row was ever captured for this date.
    missing = _latest_context(pd.Timestamp("2026-08-12").date())
    assert missing["status"] == "MARKET_CONTEXT_MISSING"
    assert missing["regime"] is None
    # Numerically, the missing case's score (0.0, the scorer's neutral
    # fallback) is identical to genuine NEUTRAL's score above -- status is
    # what makes them distinguishable, not the numbers alone.
    assert missing["score"] == available["score"] == 0.0
    assert missing["status"] != available["status"]


def test_build_observation_carries_the_market_context_status_through():
    genuinely_neutral = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=0.0,
        market_context_status="AVAILABLE",
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    missing_context = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime=None,
        market_score=0.0,
        market_context_status="MARKET_CONTEXT_MISSING",
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    assert genuinely_neutral["market_context_status"] == "AVAILABLE"
    assert missing_context["market_context_status"] == "MARKET_CONTEXT_MISSING"
    # Same market_score (0.0) either way -- the explicit status field is the
    # only reliable way to tell these two observations apart.
    assert genuinely_neutral["market_score"] == missing_context["market_score"] == 0.0
