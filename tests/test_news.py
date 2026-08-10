from datetime import datetime, timezone

from intraday_engine.market.context import score_market
from intraday_engine.market.news import NewsArticle, aggregate_news_sentiment, score_news_text
from intraday_engine.scanner.ranking import rank_candidates


def test_positive_and_negative_news_scores_are_directional():
    positive, impact_positive, high_positive = score_news_text("Company beats estimates after strong results")
    negative, impact_negative, high_negative = score_news_text("Company faces fraud investigation and guidance cut")

    assert positive > 0
    assert negative < 0
    assert impact_positive >= 0
    assert impact_negative > impact_positive
    assert not high_positive
    assert high_negative


def test_news_sentiment_contributes_bounded_market_score():
    assert score_market({"news_sentiment_score": 1.0}) == 10.0
    assert score_market({"news_sentiment_score": -1.0}) == -10.0
    assert score_market({"news_sentiment_score": 4.0}) == 10.0


def test_news_ranking_supports_bull_regime_and_penalizes_conflict():
    common = {
        "instrument_key": "NSE_EQ|TEST",
        "ltp": 100.0,
        "previous_close": 100.0,
        "cumulative_volume": 1000.0,
        "avg_daily_volume": 1000.0,
        "avg_daily_traded_value": 100_000_000.0,
        "day_high": 102.0,
        "day_low": 99.0,
        "elapsed_session_minutes": 187.5,
    }
    rows = rank_candidates(
        [
            {**common, "symbol": "GOOD", "news_score": 1.0},
            {**common, "symbol": "BAD", "news_score": -1.0},
        ],
        market_score=20.0,
        limit=2,
    )

    assert rows[0]["symbol"] == "GOOD"
    assert rows[0]["candidate_score"] > rows[1]["candidate_score"]
    assert "NEWS_SUPPORT" in rows[0]["reason"]
    assert "NEWS_CONFLICT" in rows[1]["reason"]


def test_news_aggregate_is_recency_weighted():
    now = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)
    recent = NewsArticle(
        "TEST", "KEY", "strong results", "", "recent", now, 1.0, 0.5, False
    )
    old = NewsArticle(
        "TEST", "KEY", "weak results", "", "old", now.replace(hour=0), -1.0, 0.5, False
    )

    score = aggregate_news_sentiment([recent, old], now=now)
    assert score > 0
