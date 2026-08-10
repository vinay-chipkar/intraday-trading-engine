from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import os
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketContext:
    captured_at: datetime
    gift_nifty_change_pct: float | None = None
    dow_change_pct: float | None = None
    sp500_change_pct: float | None = None
    nasdaq_change_pct: float | None = None
    india_vix: float | None = None
    usd_inr: float | None = None
    brent: float | None = None
    fii_flow: float | None = None
    dii_flow: float | None = None
    nifty_change_pct: float | None = None
    banknifty_change_pct: float | None = None
    news_count: int = 0
    high_impact_news_count: int = 0
    score: float = 0.0
    regime: str = "NEUTRAL"

    def as_dict(self) -> dict:
        return asdict(self)


def classify(score: float) -> str:
    if score >= 30:
        return "BULLISH"
    if score >= 10:
        return "MILD_BULLISH"
    if score <= -30:
        return "BEARISH"
    if score <= -10:
        return "MILD_BEARISH"
    return "NEUTRAL"


def score_market(values: dict) -> float:
    """Create a transparent regime score; weights are strategy inputs, not predictions."""
    score = 0.0
    score += (values.get("gift_nifty_change_pct") or 0.0) * 5.0
    score += (values.get("dow_change_pct") or 0.0) * 2.0
    score += (values.get("sp500_change_pct") or 0.0) * 2.0
    score += (values.get("nasdaq_change_pct") or 0.0) * 1.5
    score += (values.get("nifty_change_pct") or 0.0) * 3.0
    score += (values.get("banknifty_change_pct") or 0.0) * 2.0
    score += (values.get("fii_flow_score") or 0.0)
    score -= (values.get("india_vix_penalty") or 0.0)
    news_sentiment = max(-1.0, min(1.0, float(values.get("news_sentiment_score") or 0.0)))
    score += news_sentiment * 10.0
    return float(score)


def build_context(values: dict, timezone: str = "Asia/Kolkata") -> MarketContext:
    values = dict(values)
    score = score_market(values)
    allowed = set(MarketContext.__dataclass_fields__) - {"captured_at", "score", "regime"}
    payload = {key: values.get(key) for key in allowed}
    payload["news_count"] = int(values.get("news_count") or 0)
    payload["high_impact_news_count"] = int(values.get("high_impact_news_count") or 0)
    return MarketContext(
        captured_at=datetime.now(ZoneInfo(timezone)),
        score=score,
        regime=classify(score),
        **payload,
    )


DEFAULT_INSTRUMENT_KEYS = {
    # Upstox documents GIFT NIFTY explicitly. Global index keys can be overridden
    # through environment variables if the daily Global Instruments file changes.
    "gift_nifty": "GLOBAL_INDEX|SGX NIFTY",
    "dow": os.getenv("UPSTOX_DOW_KEY", "GLOBAL_INDEX|^DJI"),
    "sp500": os.getenv("UPSTOX_SP500_KEY", "GLOBAL_INDEX|^GSPC"),
    "nasdaq": os.getenv("UPSTOX_NASDAQ_KEY", "GLOBAL_INDEX|^IXIC"),
    "usd_inr": os.getenv("UPSTOX_USDINR_KEY", ""),
    "brent": os.getenv("UPSTOX_BRENT_KEY", "GLOBAL_INDICATOR|BZUSD"),
    "nifty": "NSE_INDEX|Nifty 50",
    "banknifty": "NSE_INDEX|Nifty Bank",
    "india_vix": "NSE_INDEX|India VIX",
}


def instrument_keys() -> dict[str, str]:
    """Return configured context instruments, excluding intentionally blank keys."""
    return {name: key for name, key in DEFAULT_INSTRUMENT_KEYS.items() if key}


def values_from_quotes(metrics: dict[str, dict]) -> dict:
    """Map normalized Upstox quote metrics to MarketContext field names."""
    keys = instrument_keys()

    def change(name: str) -> float | None:
        item = metrics.get(keys.get(name, ""), {})
        return item.get("change_pct")

    def ltp(name: str) -> float | None:
        item = metrics.get(keys.get(name, ""), {})
        return item.get("ltp")

    return {
        "gift_nifty_change_pct": change("gift_nifty"),
        "dow_change_pct": change("dow"),
        "sp500_change_pct": change("sp500"),
        "nasdaq_change_pct": change("nasdaq"),
        "nifty_change_pct": change("nifty"),
        "banknifty_change_pct": change("banknifty"),
        "india_vix": ltp("india_vix"),
        "usd_inr": ltp("usd_inr"),
        "brent": ltp("brent"),
    }


def vix_penalty(vix: float | None) -> float:
    if vix is None:
        return 0.0
    # Keep this deliberately modest; VIX is a risk-regime input, not a direction signal.
    return max(0.0, (float(vix) - 15.0) * 0.75)
