from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config.settings import settings
from intraday_engine.market.context import (
    build_context,
    instrument_keys,
    values_from_quotes,
    vix_penalty,
)
from intraday_engine.market.news import aggregate_news_sentiment, fetch_recent_news
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import (
    get_instruments,
    insert_market_context,
    insert_news,
    latest_market_news_stats,
)


def main() -> None:
    """Capture the market snapshot used by the intraday strategy."""
    client = UpstoxREST(access_token=settings.upstox_access_token)
    keys = instrument_keys()
    quotes = client.full_market_quotes(list(keys.values()))
    metrics = client.quote_metrics(quotes)

    values = values_from_quotes(metrics)
    values["india_vix_penalty"] = vix_penalty(values.get("india_vix"))
    values["fii_flow"] = None
    values["dii_flow"] = None

    now_utc = datetime.now(timezone.utc)
    instruments = get_instruments().to_dict("records")
    news_articles, news_errors = fetch_recent_news(
        client,
        instruments,
        now=now_utc,
        lookback_hours=24,
    )
    inserted_news = insert_news([article.as_db_row(now_utc) for article in news_articles])
    news_stats = latest_market_news_stats(lookback_hours=24)
    values["news_count"] = news_stats["news_count"]
    values["high_impact_news_count"] = news_stats["high_impact_news_count"]
    values["news_sentiment_score"] = aggregate_news_sentiment(news_articles, now=now_utc)

    context = build_context(values, timezone=settings.timezone)
    row = context.as_dict()
    row["trading_date"] = datetime.now(ZoneInfo(settings.timezone)).date()
    insert_market_context(row)

    print("Market context captured")
    print(f"captured_at={context.captured_at.isoformat()}")
    print(f"regime={context.regime} score={context.score:.2f}")
    print(
        f"news_count={values['news_count']} "
        f"high_impact={values['high_impact_news_count']} "
        f"news_sentiment={values['news_sentiment_score']:+.3f} "
        f"new_articles={inserted_news}"
    )
    for name, key in keys.items():
        metric = metrics.get(key, {})
        print(f"{name}: ltp={metric.get('ltp')} change_pct={metric.get('change_pct')}")
    for error in news_errors:
        print(f"NEWS_WARNING: {error}")


if __name__ == "__main__":
    main()
