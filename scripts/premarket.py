from intraday_engine.market.context import build_context
from intraday_engine.storage.db import conn
values={'gift_nifty_change_pct':None,'dow_change_pct':None,'sp500_change_pct':None,'nasdaq_change_pct':None,'india_vix':None,'usd_inr':None,'brent':None,'fii_flow':None,'dii_flow':None,'nifty_change_pct':None,'banknifty_change_pct':None,'news_count':0,'high_impact_news_count':0}
ctx=build_context(values); d=ctx.as_dict(); c=conn(); c.execute('INSERT INTO market_context VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',list(d.values())); c.close(); print(d)
