"""Causality-focused tests for research/learning_pipeline.py.

The core guarantee under test: a feature snapshot/training label for an
observation must be identical no matter what happens to candle data *after*
that observation's bar_time -- if it changes when future data changes, that's
a look-ahead leak.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.research.learning_pipeline import build_feature_snapshots_and_labels
from intraday_engine.research.paper_observer import build_observation, ensure_observation_table, persist_observations
from intraday_engine.research.paper_outcomes import ensure_outcome_table, evaluate_pending
from intraday_engine.signals.engine import TradeSignal


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "test.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)
    return fake_settings.duckdb_path


def _insert_candles(db_path: str, symbol: str, rows: list[tuple]) -> None:
    connection = db.conn(path=db_path)
    for ts, o, h, l, c, v in rows:
        connection.execute(
            f"INSERT INTO candles VALUES ('NSE_EQ|{symbol}','{symbol}', TIMESTAMPTZ '{ts}','1m',{o},{h},{l},{c},{v},NULL)"
        )
    connection.close()


def _rising_candles(symbol: str, n: int, start_minute: int = 45, step: float = 0.05, base: float = 100.0):
    rows = []
    price = base
    for i in range(n):
        minute = start_minute + i
        hour = 3 + minute // 60
        minute_of_hour = minute % 60
        ts = f"2026-08-11 {hour:02d}:{minute_of_hour:02d}:00+00"
        price += step
        rows.append((ts, price - 0.05, price + 0.2, price - 0.2, price, 1000))
    return rows


def _create_observation_and_outcome(db_path: str, symbol: str, bar_time: pd.Timestamp, entry: float) -> str:
    ensure_observation_table(db_path)
    ensure_outcome_table(db_path)
    signal = TradeSignal(
        action="BUY", score=80.0, confidence=80.0, entry=entry,
        stop_loss=entry - 1.0, target=entry + 0.3,
        reward_risk=0.3, reasons=(), blockers=(), symbol=symbol, event_time=bar_time,
    )
    row = build_observation(
        observed_at=pd.Timestamp.now(tz="UTC"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate={"symbol": symbol, "instrument_key": f"NSE_EQ|{symbol}", "rank": 1, "candidate_score": 50.0,
                   "change_pct": 1.0, "relative_volume": 1.5, "vwap": 100.0},
        signal=signal, market_regime="NEUTRAL", market_score=0.0, bar_time=bar_time,
    )
    persist_observations([row])
    return row["observation_id"]


def test_build_pipeline_produces_one_snapshot_and_label_per_evaluated_observation(isolated_db):
    rows = _rising_candles("TEST", 60)
    _insert_candles(isolated_db, "TEST", rows)
    bars = db.conn(path=isolated_db).execute("SELECT timestamp, close FROM candles ORDER BY timestamp").df()
    bar_time = bars["timestamp"].iloc[30]
    entry = float(bars["close"].iloc[30])

    _create_observation_and_outcome(isolated_db, "TEST", bar_time, entry)
    evaluate_pending(max_holding_bars=30)

    summary = build_feature_snapshots_and_labels()
    assert summary == {
        "evaluated_pending": 1, "feature_snapshots_written": 1,
        "training_labels_written": 1, "skipped_no_candles": 0,
        "from_decision_features": 0, "from_recomputed_candles": 1,
    }

    connection = db.conn(path=isolated_db)
    snapshots = connection.execute("SELECT * FROM feature_snapshots").df()
    labels = connection.execute("SELECT * FROM training_labels").df()
    connection.close()
    assert len(snapshots) == 1
    assert len(labels) == 1
    assert snapshots.iloc[0]["observation_id"] == labels.iloc[0]["observation_id"]
    assert pd.Timestamp(snapshots.iloc[0]["event_time"]) == pd.Timestamp(bar_time)


def test_pipeline_is_idempotent_on_repeated_runs(isolated_db):
    rows = _rising_candles("TEST", 60)
    _insert_candles(isolated_db, "TEST", rows)
    bars = db.conn(path=isolated_db).execute("SELECT timestamp, close FROM candles ORDER BY timestamp").df()
    bar_time = bars["timestamp"].iloc[30]
    _create_observation_and_outcome(isolated_db, "TEST", bar_time, float(bars["close"].iloc[30]))
    evaluate_pending(max_holding_bars=30)

    first = build_feature_snapshots_and_labels()
    second = build_feature_snapshots_and_labels()

    assert first["feature_snapshots_written"] == 1
    assert second["feature_snapshots_written"] == 0
    assert second["evaluated_pending"] == 0

    connection = db.conn(path=isolated_db)
    assert connection.execute("SELECT COUNT(*) FROM feature_snapshots").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM training_labels").fetchone()[0] == 1
    connection.close()


def test_pending_observation_without_an_outcome_is_never_processed(isolated_db):
    rows = _rising_candles("TEST", 60)
    _insert_candles(isolated_db, "TEST", rows)
    bars = db.conn(path=isolated_db).execute("SELECT timestamp, close FROM candles ORDER BY timestamp").df()
    bar_time = bars["timestamp"].iloc[30]
    _create_observation_and_outcome(isolated_db, "TEST", bar_time, float(bars["close"].iloc[30]))
    # deliberately do NOT call evaluate_pending -- the observation stays PENDING

    summary = build_feature_snapshots_and_labels()
    assert summary["evaluated_pending"] == 0
    assert summary["feature_snapshots_written"] == 0


def test_feature_snapshot_is_identical_regardless_of_what_happens_after_bar_time(isolated_db, tmp_path, monkeypatch):
    """The central causality guarantee: mutating/extending candles strictly
    after an observation's bar_time must not change its feature snapshot.

    Both variants share identical bars 0..30 (decision bar = index 30) and
    identical "resolution" bars 31..35 (so the trade evaluates to the same
    outcome in both cases, and the observation actually reaches EVALUATED).
    They differ only in a wild, discontinuous block appended *after*
    resolution has already happened -- data the snapshot query's
    `timestamp <= bar_time` filter must never see, and data the evaluator
    never reaches either since it already exited during the resolution bars.
    """

    def build_and_snapshot(extra_future_rows: list[tuple], suffix: str) -> dict:
        fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / f"db_{suffix}.duckdb"))
        monkeypatch.setattr(db, "settings", fake_settings)
        path = fake_settings.duckdb_path

        base_rows = _rising_candles("TEST", 31)  # bars 0..30, decision bar is index 30
        resolution_rows = _rising_candles("TEST", 5, start_minute=76, step=1.0, base=base_rows[-1][4])
        _insert_candles(path, "TEST", base_rows + resolution_rows + extra_future_rows)

        bars = db.conn(path=path).execute("SELECT timestamp, close FROM candles ORDER BY timestamp").df()
        bar_time = bars["timestamp"].iloc[30]
        _create_observation_and_outcome(path, "TEST", bar_time, float(bars["close"].iloc[30]))
        evaluation = evaluate_pending(max_holding_bars=5)
        assert evaluation["evaluated"] == 1  # sanity: both variants must actually resolve

        summary = build_feature_snapshots_and_labels()
        assert summary["feature_snapshots_written"] == 1

        connection = db.conn(path=path)
        snapshot = connection.execute(
            "SELECT * EXCLUDE (observation_id) FROM feature_snapshots"
        ).df().to_dict("records")[0]
        connection.close()
        return snapshot

    no_extra_future = build_and_snapshot([], "plain")
    # A wild, discontinuous future move that would change every rolling
    # indicator (EMA/RSI/ATR/support/resistance/VWAP) if it leaked backward.
    wild_future = build_and_snapshot(
        _rising_candles("TEST", 20, start_minute=200, step=50.0, base=100.0), "wild"
    )

    # NaN-tolerant comparison: ema50/ema200 are legitimately NaN this early in
    # the series (fewer than 50/200 bars exist yet) in both variants, and
    # NaN != NaN under plain dict equality.
    pd.testing.assert_series_equal(pd.Series(no_extra_future), pd.Series(wild_future))


def test_training_label_matches_the_already_computed_paper_outcome_not_a_recomputation(isolated_db):
    rows = _rising_candles("TEST", 60)
    _insert_candles(isolated_db, "TEST", rows)
    bars = db.conn(path=isolated_db).execute("SELECT timestamp, close FROM candles ORDER BY timestamp").df()
    bar_time = bars["timestamp"].iloc[30]
    _create_observation_and_outcome(isolated_db, "TEST", bar_time, float(bars["close"].iloc[30]))
    evaluate_pending(max_holding_bars=30)

    connection = db.conn(path=isolated_db)
    outcome_row = connection.execute("SELECT outcome, mfe_points, mae_points FROM paper_outcomes").df().iloc[0]
    connection.close()

    build_feature_snapshots_and_labels()

    connection = db.conn(path=isolated_db)
    label_row = connection.execute("SELECT * FROM training_labels").df().iloc[0]
    connection.close()

    assert label_row["target_hit_first"] == (outcome_row["outcome"] in {"TARGET", "TARGET_GAP"})
    assert label_row["label"] == int(outcome_row["outcome"] in {"TARGET", "TARGET_GAP"})
    assert label_row["max_favorable_excursion"] == pytest.approx(outcome_row["mfe_points"])
    assert label_row["max_adverse_excursion"] == pytest.approx(outcome_row["mae_points"])


def test_stop_outcome_is_labeled_zero_not_one(isolated_db):
    # A falling price series so the stop, not the target, is hit first.
    rows = []
    price = 100.0
    for i in range(60):
        minute = 45 + i
        hour = 3 + minute // 60
        minute_of_hour = minute % 60
        ts = f"2026-08-11 {hour:02d}:{minute_of_hour:02d}:00+00"
        price -= 0.05
        rows.append((ts, price + 0.05, price + 0.2, price - 0.2, price, 1000))
    _insert_candles(isolated_db, "TEST", rows)
    bars = db.conn(path=isolated_db).execute("SELECT timestamp, close FROM candles ORDER BY timestamp").df()
    bar_time = bars["timestamp"].iloc[30]
    _create_observation_and_outcome(isolated_db, "TEST", bar_time, float(bars["close"].iloc[30]))
    evaluate_pending(max_holding_bars=30)

    build_feature_snapshots_and_labels()
    connection = db.conn(path=isolated_db)
    label_row = connection.execute("SELECT * FROM training_labels").df().iloc[0]
    outcome = connection.execute("SELECT outcome FROM paper_outcomes").df().iloc[0]["outcome"]
    connection.close()

    assert outcome in {"STOP", "STOP_GAP"}
    assert label_row["label"] == 0
    assert label_row["target_hit_first"] == False  # noqa: E712


def test_observation_with_no_matching_candle_is_skipped_not_guessed(isolated_db):
    ensure_observation_table(isolated_db)
    ensure_outcome_table(isolated_db)
    # An observation whose bar_time has no corresponding candle row at all
    # (e.g. candles were pruned/never ingested for that exact minute).
    bar_time = pd.Timestamp("2026-08-11 04:15:00+00:00")
    obs_id = _create_observation_and_outcome(isolated_db, "GHOST", bar_time, 100.0)

    connection = db.conn(path=isolated_db)
    connection.execute(
        """
        INSERT INTO paper_outcomes VALUES
        (?, now(), TIMESTAMPTZ '2026-08-11 04:16:00+00', TIMESTAMPTZ '2026-08-11 04:20:00+00',
         'LONG', 100.0, 100.3, 99.0, 100.3, 'TARGET', 0.3, 0.3, 4, 0.3, -0.1, '1.0.0')
        """,
        [obs_id],
    )
    connection.execute("UPDATE paper_observations SET status = 'EVALUATED' WHERE observation_id = ?", [obs_id])
    connection.close()

    summary = build_feature_snapshots_and_labels()
    assert summary["evaluated_pending"] == 1
    assert summary["feature_snapshots_written"] == 0
    assert summary["skipped_no_candles"] == 1


def _create_real_observation_via_signal_path(db_path: str, symbol: str) -> tuple[str, dict, pd.Timestamp]:
    """Builds an observation through the real _signal_for_symbol/build_observation
    path (decision_features captured), rather than test helper _create_observation_and_outcome's
    hand-built TradeSignal (which never captures decision_features).

    _signal_for_symbol always scores the *latest* available candle, so the
    caller must insert only the "decision" history first and add resolution
    bars afterward (mirroring how the real pipeline ticks forward in time).
    """
    from intraday_engine.research.paper_observer import _signal_for_symbol

    ensure_observation_table(db_path)
    ensure_outcome_table(db_path)
    signal, bar_time, decision_features = _signal_for_symbol(
        symbol, instrument_key=f"NSE_EQ|{symbol}", market_score=0.0, min_score=1.0
    )
    row = build_observation(
        observed_at=pd.Timestamp.now(tz="UTC"),
        trading_date=pd.Timestamp("2026-08-11").date(),
        candidate={"symbol": symbol, "instrument_key": f"NSE_EQ|{symbol}", "rank": 1, "candidate_score": 50.0,
                   "change_pct": 1.0, "relative_volume": 1.5, "vwap": 100.0},
        signal=signal, market_regime="NEUTRAL", market_score=0.0, bar_time=bar_time,
        decision_features=decision_features,
    )
    persist_observations([row])
    return row["observation_id"], decision_features, pd.Timestamp(bar_time)


def test_feature_snapshot_uses_the_captured_decision_features_not_a_recomputation(isolated_db):
    _insert_candles(isolated_db, "TEST", _rising_candles("TEST", 31))  # decision history, bar 30 is "now"
    obs_id, decision_features, bar_time = _create_real_observation_via_signal_path(isolated_db, "TEST")
    # resolution bars, strictly after the decision bar
    _insert_candles(isolated_db, "TEST", _rising_candles("TEST", 5, start_minute=76))
    evaluate_pending(max_holding_bars=5)

    summary = build_feature_snapshots_and_labels()
    assert summary["from_decision_features"] == 1
    assert summary["from_recomputed_candles"] == 0

    connection = db.conn(path=isolated_db)
    snapshot = connection.execute(
        "SELECT trend, relative_volume, rsi14, event_time FROM feature_snapshots WHERE observation_id = ?", [obs_id]
    ).df().iloc[0]
    connection.close()

    assert pd.Timestamp(snapshot["event_time"]) == bar_time
    assert snapshot["trend"] == decision_features["trend"]
    if decision_features["relative_volume"] is not None:
        assert snapshot["relative_volume"] == pytest.approx(decision_features["relative_volume"])


def test_feature_snapshot_is_unaffected_by_a_candle_revision_made_after_capture(isolated_db):
    """The decisive proof for why decision_features must be captured, not
    recomputed: if a candle used at signal time is later revised (e.g.
    market/ingestion.py::upsert_candles correcting a provisional bar), the
    feature snapshot must still reflect what the signal actually saw, not
    the revised data.
    """
    _insert_candles(isolated_db, "TEST", _rising_candles("TEST", 31))
    obs_id, decision_features, bar_time = _create_real_observation_via_signal_path(isolated_db, "TEST")
    _insert_candles(isolated_db, "TEST", _rising_candles("TEST", 5, start_minute=76))
    evaluate_pending(max_holding_bars=5)

    # Revise every stored candle's close/volume in place (same timestamps,
    # different values) -- simulates upstream correcting provisional bars
    # *after* the signal was already generated from the original values.
    connection = db.conn(path=isolated_db)
    connection.execute(
        "UPDATE candles SET close = close * 3, volume = volume * 5 WHERE instrument_key = 'NSE_EQ|TEST'"
    )
    connection.close()

    build_feature_snapshots_and_labels()

    connection = db.conn(path=isolated_db)
    snapshot = connection.execute(
        "SELECT trend, rsi14, close FROM feature_snapshots WHERE observation_id = ?", [obs_id]
    ).df().iloc[0]
    connection.close()

    # Matches the ORIGINAL captured decision, not a recomputation against
    # the now-revised (3x close, 5x volume) candles.
    assert snapshot["close"] == pytest.approx(decision_features["close"])
    assert snapshot["trend"] == decision_features["trend"]
