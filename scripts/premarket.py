from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from config.settings import settings
from intraday_engine.market.context import (
    build_context,
    instrument_keys,
    values_from_quotes,
    vix_penalty,
)
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import insert_market_context


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
    values["news_count"] = 0
    values["high_impact_news_count"] = 0

    context = build_context(values, timezone=settings.timezone)
    row = context.as_dict()
    row["trading_date"] = datetime.now(ZoneInfo(settings.timezone)).date()
    insert_market_context(row)

    print("Market context captured")
    print(f"captured_at={context.captured_at.isoformat()}")
    print(f"regime={context.regime} score={context.score:.2f}")
    for name, key in keys.items():
        metric = metrics.get(key, {})
        print(f"{name}: ltp={metric.get('ltp')} change_pct={metric.get('change_pct')}")


if __name__ == "__main__":
    main()
