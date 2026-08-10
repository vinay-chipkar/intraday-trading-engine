import pandas as pd
import numpy as np
from intraday_engine.technical.indicators import add_indicators

def test_indicators():
    ts=pd.date_range('2026-01-01 09:15',periods=80,freq='min',tz='Asia/Kolkata'); close=np.linspace(100,120,80)
    df=pd.DataFrame({'timestamp':ts,'open':close-.2,'high':close+.5,'low':close-.5,'close':close,'volume':np.arange(1,81)*1000,'open_interest':0})
    o=add_indicators(df); assert {'ema9','ema20','ema50','rsi14','atr14','vwap','relative_volume'}<=set(o.columns); assert o.vwap.notna().any()
