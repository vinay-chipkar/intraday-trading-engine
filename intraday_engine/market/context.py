from dataclasses import dataclass, asdict
from datetime import datetime
from zoneinfo import ZoneInfo

@dataclass
class MarketContext:
    captured_at: datetime
    gift_nifty_change_pct: float|None=None
    dow_change_pct: float|None=None
    sp500_change_pct: float|None=None
    nasdaq_change_pct: float|None=None
    india_vix: float|None=None
    usd_inr: float|None=None
    brent: float|None=None
    fii_flow: float|None=None
    dii_flow: float|None=None
    nifty_change_pct: float|None=None
    banknifty_change_pct: float|None=None
    news_count: int=0
    high_impact_news_count: int=0
    score: float=0.0
    regime: str="NEUTRAL"
    def as_dict(self): return asdict(self)

def classify(score):
    if score>=30:return "BULLISH"
    if score>=10:return "MILD_BULLISH"
    if score<=-30:return "BEARISH"
    if score<=-10:return "MILD_BEARISH"
    return "NEUTRAL"

def build_context(v, timezone="Asia/Kolkata"):
    score=(v.get("gift_nifty_change_pct") or 0)*5+(v.get("dow_change_pct") or 0)*2+(v.get("sp500_change_pct") or 0)*2+(v.get("nasdaq_change_pct") or 0)+(v.get("nifty_change_pct") or 0)*3+(v.get("banknifty_change_pct") or 0)*2+(v.get("fii_flow_score") or 0)-(v.get("india_vix_penalty") or 0)
    return MarketContext(datetime.now(ZoneInfo(timezone)), score=float(score), regime=classify(score), **{k:v.get(k) for k in MarketContext.__dataclass_fields__ if k not in {"captured_at","score","regime"}})
