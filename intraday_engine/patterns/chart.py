from dataclasses import dataclass
import pandas as pd
from intraday_engine.technical.structure import swing_points
@dataclass
class Pattern:
    name:str; confidence:float; neckline:float|None=None; level:float|None=None
def double_top_bottom(df,window=3,tolerance=.012):
    sh,sl=swing_points(df,window); hs=df.loc[sh,['timestamp','high']].tail(8); ls=df.loc[sl,['timestamp','low']].tail(8); out=[]
    if len(hs)>=2:
        a,b=hs.iloc[-2],hs.iloc[-1]
        if abs(a.high-b.high)/max(a.high,b.high)<=tolerance:
            x=df[(df.timestamp>a.timestamp)&(df.timestamp<b.timestamp)]; out.append(Pattern('DOUBLE_TOP',80,float(x.low.min()),float((a.high+b.high)/2))) if not x.empty else None
    if len(ls)>=2:
        a,b=ls.iloc[-2],ls.iloc[-1]
        if abs(a.low-b.low)/max(a.low,b.low)<=tolerance:
            x=df[(df.timestamp>a.timestamp)&(df.timestamp<b.timestamp)]; out.append(Pattern('DOUBLE_BOTTOM',80,float(x.high.max()),float((a.low+b.low)/2))) if not x.empty else None
    return out
