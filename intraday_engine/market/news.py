from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Iterable

from intraday_engine.market.upstox import UpstoxREST


POSITIVE_TERMS = {
    "beat": 1.0,
    "beats": 1.0,
    "beating": 1.0,
    "strong results": 1.0,
    "profit rises": 1.0,
    "profit jumped": 1.0,
    "profit surges": 1.0,
    "revenue rises": 0.8,
    "order win": 1.0,
    "order wins": 1.0,
    "contract win": 1.0,
    "major order": 0.8,
    "upgrade": 0.9,
    "upgraded": 0.9,
    "buyback": 0.8,
    "dividend": 0.6,
    "partnership": 0.6,
    "expansion": 0.5,
    "guidance raised": 1.0,
    "outlook improved": 0.8,
    "record revenue": 1.0,
    "record profit": 1.0,
}

NEGATIVE_TERMS = {
    "miss": 1.0,
    "misses": 1.0,
    "missed estimates": 1.0,
    "weak results": 1.0,
    "profit falls": 1.0,
    "profit fell": 1.0,
    "profit declines": 1.0,
    "loss widens": 1.0,
    "downgrade": 0.9,
    "downgraded": 0.9,
    "guidance cut": 1.0,
    "outlook cut": 0.8,
    "investigation": 1.0,
    "penalty": 0.8,
    "fraud": 1.2,
    "default": 1.2,
    "regulatory action": 1.0,
    "resignation": 0.6,
    "plant shutdown": 0.9,
    "order cancelled": 1.0,
    "order cancellation": 1.0,
}

HIGH_IMPACT_TERMS = {
    "fraud",
    "default",
    "investigation",
    "regulatory action",
    "penalty",
    "guidance cut",
    "guidance raised",
    "profit falls",
    "profit fell",
    "profit surges",
    "major order",
    "order cancelled",
    "order cancellation",
}


def _score_terms(text: str, terms: dict[str, float]) -> float:
    return sum(weight for term, weight in terms.items() if term in text)


def score_news_text(text: str) -> tuple[float, float, bool]:
    """Return sentiment [-1, 1], impact [0, 1], and high-impact flag.

    This is intentionally a transparent baseline, not an ML sentiment model.
    """
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    positive = _score_terms(normalized, POSITIVE_TERMS)
    negative = _score_terms(normalized, NEGATIVE_TERMS)
    total = positive + negative
    sentiment = 0.0 if total == 0 else (positive - negative) / max(total, 1.0)
    sentiment = max(-1.0, min(1.0, sentiment))
    high_impact = any(term in normalized for term in HIGH_IMPACT_TERMS)
    impact = min(1.0, abs(sentiment) * 0.7 + (0.3 if high_impact else 0.0))
    return round(sentiment, 6), round(impact, 6), high_impact


@dataclass(frozen=True)
class NewsArticle:
    symbol: str
    instrument_key: str
    heading: str
    summary: str
    article_link: str
    published_at: datetime
    sentiment_score: float
    impact_score: float
    high_impact: bool

    def as_db_row(self, captured_at: datetime) -> dict:
        return {
            "captured_at": captured_at,
            "published_at": self.published_at,
            "symbol": self.symbol,
            "instrument_key": self.instrument_key,
            "heading": self.heading,
            "summary": self.summary,
            "article_link": self.article_link,
            "sentiment_score": self.sentiment_score,
            "impact_score": self.impact_score,
            "high_impact": self.high_impact,
        }


def _published_at(value: object) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def fetch_recent_news(
    client: UpstoxREST,
    instruments: Iterable[dict],
    *,
    now: datetime | None = None,
    lookback_hours: int = 24,
    batch_size: int = 30,
) -> tuple[list[NewsArticle], list[str]]:
    """Fetch and score recent instrument news, respecting Upstox's 30-key limit."""
    if lookback_hours < 1:
        raise ValueError("lookback_hours must be >= 1")
    if not 1 <= batch_size <= 30:
        raise ValueError("batch_size must be between 1 and 30")

    captured_at = now or datetime.now(timezone.utc)
    cutoff = captured_at - timedelta(hours=lookback_hours)
    rows = [dict(row) for row in instruments]
    articles: list[NewsArticle] = []
    errors: list[str] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        keys = [str(row.get("instrument_key", "")) for row in batch if row.get("instrument_key")]
        if not keys:
            continue
        try:
            payload = client.news(keys, page_size=100)
        except Exception as exc:  # noqa: BLE001 - premarket should retain partial coverage
            errors.append(f"news batch {start // batch_size + 1}: {exc}")
            continue

        symbol_by_key = {str(row["instrument_key"]): str(row["symbol"]) for row in batch}
        for key, raw_items in payload.items():
            symbol = symbol_by_key.get(str(key))
            if not symbol or not isinstance(raw_items, list):
                continue
            for raw in raw_items:
                if not isinstance(raw, dict):
                    continue
                published_at = _published_at(raw.get("published_time"))
                if published_at is None or published_at < cutoff or published_at > captured_at + timedelta(minutes=5):
                    continue
                heading = str(raw.get("heading") or "").strip()
                summary = str(raw.get("summary") or "").strip()
                link = str(raw.get("article_link") or "").strip()
                if not heading and not summary:
                    continue
                sentiment, impact, high_impact = score_news_text(f"{heading} {summary}")
                articles.append(
                    NewsArticle(
                        symbol=symbol,
                        instrument_key=str(key),
                        heading=heading,
                        summary=summary,
                        article_link=link,
                        published_at=published_at,
                        sentiment_score=sentiment,
                        impact_score=impact,
                        high_impact=high_impact,
                    )
                )

    deduped: dict[tuple[str, int, str], NewsArticle] = {}
    for article in articles:
        key = (article.instrument_key, int(article.published_at.timestamp() * 1000), article.article_link)
        deduped[key] = article
    return list(deduped.values()), errors


def aggregate_news_sentiment(
    articles: Iterable[NewsArticle],
    *,
    now: datetime | None = None,
) -> float:
    """Recency-weighted market news sentiment in [-1, 1]."""
    captured_at = now or datetime.now(timezone.utc)
    weighted_sum = 0.0
    weight_sum = 0.0
    for article in articles:
        age_hours = max(0.0, (captured_at - article.published_at).total_seconds() / 3600.0)
        recency = math.exp(-age_hours / 12.0)
        weight = recency * (1.0 + article.impact_score)
        weighted_sum += article.sentiment_score * weight
        weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return round(max(-1.0, min(1.0, weighted_sum / weight_sum)), 6)
