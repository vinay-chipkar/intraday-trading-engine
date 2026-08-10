import numpy as np
import pandas as pd

def ema(s, span): return s.ewm(span=span, adjust=False, min_periods=span).mean()
def rsi(s, period=14):
    d=s.diff(); g=d.clip(lower=0).ewm(alpha=1/period,adjust=False,min_periods=period).mean(); l=(-d.clip(upper=0)).ewm(alpha=1/period,adjust=False,min_periods=period).mean(); return (100-100/(1+g/l.replace(0,np.nan))).fillna(50)
def atr(df, period=14):
    p=df.close.shift(1); tr=pd.concat([df.high-df.low,(df.high-p).abs(),(df.low-p).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
def vwap(df):
    s=df.timestamp.dt.date; tp=(df.high+df.low+df.close)/3; pv=tp*df.volume; return pv.groupby(s).cumsum()/df.volume.groupby(s).cumsum().replace(0,np.nan)
def add_indicators(df):
    o=df.copy().sort_values("timestamp").reset_index(drop=True); o["ema9"]=ema(o.close,9); o["ema20"]=ema(o.close,20); o["ema50"]=ema(o.close,50); o["ema200"]=ema(o.close,200); o["rsi14"]=rsi(o.close); o["atr14"]=atr(o); o["vwap"]=vwap(o); o["relative_volume"]=o.volume/o.volume.rolling(20,min_periods=5).mean().replace(0,np.nan); return o
