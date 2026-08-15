import dataclasses
import json

import pandas as pd
import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.research.paper_observer import (
    _decision_observation_id,
    _latest_context,
    _load_bars,
    _signal_for_symbol,
    build_observation,
    is_bar_stale,
    is_bar_stale_intraday,
    persist_observations,
)
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


def test_is_bar_stale_intraday_true_when_bar_is_older_than_max_age():
    now = pd.Timestamp("2026-08-13 06:00:00+00:00")
    bar_time = pd.Timestamp("2026-08-13 05:40:00+00:00")  # 20 minutes old
    assert is_bar_stale_intraday(bar_time, now, max_age_minutes=15)


def test_is_bar_stale_intraday_false_within_max_age():
    now = pd.Timestamp("2026-08-13 06:00:00+00:00")
    bar_time = pd.Timestamp("2026-08-13 05:50:00+00:00")  # 10 minutes old
    assert not is_bar_stale_intraday(bar_time, now, max_age_minutes=15)


def test_is_bar_stale_intraday_exactly_at_the_boundary_is_not_stale():
    now = pd.Timestamp("2026-08-13 06:00:00+00:00")
    bar_time = pd.Timestamp("2026-08-13 05:45:00+00:00")  # exactly 15 minutes old
    assert not is_bar_stale_intraday(bar_time, now, max_age_minutes=15)


def test_is_bar_stale_intraday_false_for_naive_timestamps_rather_than_crashing():
    assert not is_bar_stale_intraday(pd.Timestamp("2026-08-13 05:40:00"), pd.Timestamp("2026-08-13 06:00:00"))


def test_intraday_stale_bar_forces_stale_status_with_a_transparent_reason():
    # Same trading day, but the bar is 20 minutes old at observation time --
    # ingestion has clearly stalled mid-session, even though this isn't
    # "yesterday's" data (is_bar_stale alone would say "not stale").
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-13 06:00:00+00:00"),
        trading_date=pd.Timestamp("2026-08-13").date(),
        candidate=_candidate(),
        signal=_signal("SELL"),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-13 05:40:00+00:00"),
        stale_reason="INTRADAY_STALE",
    )
    assert row["status"] == "STALE_DATA"
    assert row["stale_reason"] == "INTRADAY_STALE"
    assert row["entry_price"] is None


def test_prior_day_and_intraday_stale_reasons_are_distinguishable():
    prior_day = build_observation(
        observed_at=pd.Timestamp("2026-08-13 06:00:00+00:00"),
        trading_date=pd.Timestamp("2026-08-13").date(),
        candidate=_candidate(),
        signal=_signal("SELL"),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-12 05:40:00+00:00"),
        stale_reason="PRIOR_DAY",
    )
    intraday = build_observation(
        observed_at=pd.Timestamp("2026-08-13 06:00:00+00:00"),
        trading_date=pd.Timestamp("2026-08-13").date(),
        candidate=_candidate(),
        signal=_signal("SELL"),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-13 05:40:00+00:00"),
        stale_reason="INTRADAY_STALE",
    )
    fresh = build_observation(
        observed_at=pd.Timestamp("2026-08-13 06:00:00+00:00"),
        trading_date=pd.Timestamp("2026-08-13").date(),
        candidate=_candidate(),
        signal=_signal("SELL"),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-13 05:58:00+00:00"),
    )
    # All three would look identical if only `status` were inspected --
    # stale_reason is what makes them distinguishable for diagnostics.
    assert prior_day["status"] == intraday["status"] == "STALE_DATA"
    assert fresh["status"] == "SIGNAL_PENDING"
    assert {prior_day["stale_reason"], intraday["stale_reason"], fresh["stale_reason"]} == {
        "PRIOR_DAY", "INTRADAY_STALE", None,
    }


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


# --- decision-time feature capture: the exact row generate_signal used must
# be what gets stored, not a value re-derived later by a different engine ---


def _insert_rising_candles(db_path: str, symbol: str, n: int = 60) -> None:
    connection = db.conn(path=db_path)
    price = 100.0
    for i in range(n):
        minute = 45 + i
        hour = 3 + minute // 60
        minute_of_hour = minute % 60
        ts = f"2026-08-13 {hour:02d}:{minute_of_hour:02d}:00+00"
        price += 0.05
        connection.execute(
            f"INSERT INTO candles VALUES ('NSE_EQ|{symbol}', '{symbol}', TIMESTAMPTZ '{ts}', '1m', "
            f"{price - 0.05}, {price + 0.2}, {price - 0.2}, {price}, 1000, NULL)"
        )
    connection.close()


def test_signal_for_symbol_captures_the_exact_row_it_scored(isolated_db):
    _insert_rising_candles(isolated_db, "AAA", n=60)
    signal, bar_time, decision_features = _signal_for_symbol("AAA", instrument_key="NSE_EQ|AAA", market_score=0.0, min_score=1.0)

    assert decision_features["symbol"] == "AAA"
    assert decision_features["instrument_key"] == "NSE_EQ|AAA"
    assert pd.Timestamp(decision_features["event_time"]) == pd.Timestamp(bar_time)
    # trend/relative_volume are exactly the fields signals/engine.py::_score
    # read off this same row to build the signal -- must be present and
    # numeric/string, not silently dropped or substituted.
    assert decision_features["trend"] in {"UPTREND", "DOWNTREND", "SIDEWAYS"}
    assert isinstance(decision_features["relative_volume"], (int, float)) or decision_features["relative_volume"] is None


def test_decision_features_round_trip_through_build_observation_as_json(isolated_db):
    _insert_rising_candles(isolated_db, "AAA", n=60)
    signal, bar_time, decision_features = _signal_for_symbol("AAA", instrument_key="NSE_EQ|AAA", market_score=0.0, min_score=1.0)

    row = build_observation(
        observed_at=pd.Timestamp("2026-08-13 10:00:00+05:30"),
        trading_date=pd.Timestamp("2026-08-13").date(),
        candidate=_candidate(),
        signal=signal,
        market_regime="NEUTRAL",
        market_score=0.0,
        bar_time=bar_time,
        decision_features=decision_features,
    )
    stored = json.loads(row["decision_features"])
    assert stored["trend"] == decision_features["trend"]
    assert stored["symbol"] == "AAA"


def test_build_observation_decision_features_is_none_when_not_supplied():
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=0.0,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    assert row["decision_features"] is None


# --- instrument_key remapping: a symbol-only query must never mix two
# different instrument_keys' candle histories together ---


def test_load_bars_does_not_mix_histories_across_a_remapped_instrument_key(isolated_db):
    connection = db.conn(path=isolated_db)
    # Old listing: 3 candles at price ~100. New listing (same ticker,
    # renamed/relisted under a new instrument_key): 2 candles at price ~500 --
    # a real discretionary trader would never treat these as one series.
    for i, price in enumerate([100.0, 100.5, 101.0]):
        connection.execute(
            f"INSERT INTO candles VALUES ('NSE_EQ|OLD_KEY', 'ZOMATO', "
            f"TIMESTAMPTZ '2026-08-10 0{3+i}:45:00+00', '1m', {price}, {price+1}, {price-1}, {price}, 1000, NULL)"
        )
    for i, price in enumerate([500.0, 501.0]):
        connection.execute(
            f"INSERT INTO candles VALUES ('NSE_EQ|NEW_KEY', 'ZOMATO', "
            f"TIMESTAMPTZ '2026-08-13 0{3+i}:45:00+00', '1m', {price}, {price+1}, {price-1}, {price}, 1000, NULL)"
        )
    connection.close()

    old_bars = _load_bars("NSE_EQ|OLD_KEY")
    new_bars = _load_bars("NSE_EQ|NEW_KEY")

    assert len(old_bars) == 3
    assert len(new_bars) == 2
    assert old_bars["close"].max() < 200  # old listing's price range only
    assert new_bars["close"].min() > 400  # new listing's price range only


# --- idempotency: a replayed workflow tick must not create a second,
# duplicate economic observation for the same decision bar/version ---


def test_decision_observation_id_is_deterministic_for_the_same_decision():
    first = _decision_observation_id(
        instrument_key="NSE_EQ|TEST",
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
        strategy_version="1.0.0",
        feature_engine_version="1.0.0",
    )
    replay = _decision_observation_id(
        instrument_key="NSE_EQ|TEST",
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
        strategy_version="1.0.0",
        feature_engine_version="1.0.0",
    )
    assert first == replay


def test_decision_observation_id_differs_across_bar_time_instrument_or_version():
    base = dict(
        instrument_key="NSE_EQ|TEST",
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
        strategy_version="1.0.0",
        feature_engine_version="1.0.0",
    )
    baseline = _decision_observation_id(**base)
    different_bar = _decision_observation_id(**{**base, "bar_time": pd.Timestamp("2026-08-11 04:56:00+00:00")})
    different_instrument = _decision_observation_id(**{**base, "instrument_key": "NSE_EQ|OTHER"})
    different_strategy_version = _decision_observation_id(**{**base, "strategy_version": "2.0.0"})
    different_feature_version = _decision_observation_id(**{**base, "feature_engine_version": "2.0.0"})

    assert len({baseline, different_bar, different_instrument, different_strategy_version, different_feature_version}) == 5


def test_build_observation_reuses_the_same_id_for_the_same_decision():
    kwargs = dict(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    first_tick = build_observation(**kwargs)
    # A later, independent call for the exact same decision bar -- as would
    # happen if a workflow tick were replayed/retried.
    replayed_tick = build_observation(**kwargs)
    assert first_tick["observation_id"] == replayed_tick["observation_id"]


def test_persist_observations_drops_a_replayed_observation_instead_of_duplicating_it(isolated_db):
    row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )

    first_inserted = persist_observations([row])
    # Simulate a replayed workflow tick recomputing the identical decision
    # (same instrument_key/bar_time/versions) and trying to persist it again.
    replay_inserted = persist_observations([dict(row)])

    assert first_inserted == 1
    assert replay_inserted == 0

    connection = db.conn()
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM paper_observations WHERE observation_id = ?",
            [row["observation_id"]],
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 1


def test_persist_observations_still_inserts_a_genuinely_new_decision(isolated_db):
    first_row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:25:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 04:55:00+00:00"),
    )
    # A later tick against a genuinely new bar -- must not be treated as a replay.
    second_row = build_observation(
        observed_at=pd.Timestamp("2026-08-11 10:30:00+05:30"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate=_candidate(),
        signal=_signal(),
        market_regime="NEUTRAL",
        market_score=1.07,
        bar_time=pd.Timestamp("2026-08-11 05:00:00+00:00"),
    )

    assert persist_observations([first_row]) == 1
    assert persist_observations([second_row]) == 1
    assert first_row["observation_id"] != second_row["observation_id"]
