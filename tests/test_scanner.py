from intraday_engine.scanner.ranking import rank_candidates


def row(symbol, change, rvol, liquidity, day_range=2.0):
    return {
        "symbol": symbol,
        "instrument_key": f"NSE_EQ|{symbol}",
        "ltp": 100.0,
        "previous_close": 100.0,
        "change_pct": change,
        "cumulative_volume": rvol * 1000,
        "avg_daily_volume": 1000.0,
        "avg_daily_traded_value": liquidity,
        "day_high": 100.0 + day_range / 2,
        "day_low": 100.0 - day_range / 2,
        "elapsed_session_minutes": 375.0,
        "history_days": 20,
        "required_history_days": 20,
    }


def test_ranker_returns_top_n_and_is_deterministic():
    rows = [
        row("AAA", 2.0, 2.5, 200_000_000),
        row("BBB", 1.0, 1.2, 100_000_000),
        row("CCC", -1.5, 2.0, 150_000_000),
    ]
    first = rank_candidates(rows, market_score=10, limit=2)
    second = rank_candidates(rows, market_score=10, limit=2)
    assert [r["symbol"] for r in first] == [r["symbol"] for r in second]
    assert len(first) == 2
    assert first[0]["rank"] == 1
    assert first[0]["candidate_score"] > first[1]["candidate_score"]


def test_ranker_can_return_full_universe_without_truncation():
    rows = [
        row("AAA", 2.0, 2.5, 200_000_000),
        row("BBB", 1.0, 1.2, 100_000_000),
        row("CCC", -1.5, 2.0, 150_000_000),
    ]
    ranked = rank_candidates(rows, market_score=0, limit=None)
    assert len(ranked) == 3
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert {r["symbol"] for r in ranked} == {"AAA", "BBB", "CCC"}


def test_rvol_is_time_adjusted():
    rows = [row("AAA", 1.0, 2.0, 100_000_000)]
    ranked = rank_candidates(rows, market_score=0, limit=1)
    assert ranked[0]["relative_volume"] == 2.0


def test_bull_market_rewards_positive_momentum():
    rows = [
        row("UP", 2.0, 1.5, 100_000_000),
        row("DOWN", -2.0, 1.5, 100_000_000),
    ]
    ranked = rank_candidates(rows, market_score=10, limit=2)
    assert ranked[0]["symbol"] == "UP"
