import pandas as pd
from config import settings
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.strategy.features import enrich
from intraday_engine.strategy.scoring import score_long,score_short
from intraday_engine.storage.db import conn,insert_df,latest_market_score

def scan():
    api=UpstoxREST(settings.upstox_access_token); c=conn(); universe=c.execute('SELECT symbol,instrument_key FROM instrument_master').fetchall(); c.close(); rows=[]; market=latest_market_score()
    for symbol,key in universe:
        try: df=api.intraday_candles(key,'minutes',settings.candle_interval)
        except Exception as e: print(f'WARN {symbol}: {e}'); continue
        if len(df)<60: continue
        f=enrich(df); r=f.iloc[-1]; ls,lr=score_long(r,market); ss,sr=score_short(r,market); score,reasons,side=(ls,lr,'LONG') if ls>=ss else (ss,sr,'SHORT')
        rows.append({'event_time':r.timestamp,'trading_date':r.timestamp.date(),'symbol':symbol,'instrument_key':key,'ltp':r.close,'volume':r.volume,'relative_volume':r.relative_volume,'price_change_pct':(r.close/f.iloc[0].open-1)*100,'vwap':r.vwap,'candidate_score':score,'reason':side+':'+','.join(reasons)})
    out=pd.DataFrame(rows)
    if not out.empty: out=out.sort_values('candidate_score',ascending=False).head(settings.top_n); insert_df('candidate_events',out)
    return out
if __name__=='__main__':
    result=scan(); print(result.to_string(index=False) if not result.empty else 'No candidates')
