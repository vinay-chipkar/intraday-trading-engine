import pandas as pd

def swing_points(df, window=3):
    return (df.high==df.high.rolling(2*window+1,center=True).max()).fillna(False),(df.low==df.low.rolling(2*window+1,center=True).min()).fillna(False)
def support_resistance(df, window=3, cluster_pct=.003):
    sh,sl=swing_points(df,window); price=float(df.close.iloc[-1]); highs=df.loc[sh,'high'].tolist(); lows=df.loc[sl,'low'].tolist(); sup=[x for x in lows if x<=price*(1+cluster_pct)]; res=[x for x in highs if x>=price*(1-cluster_pct)]; return (max(sup) if sup else None),(min(res) if res else None)
def trend_from_swings(df,window=3):
    sh,sl=swing_points(df,window); h=df.loc[sh,'high'].tail(3).tolist(); l=df.loc[sl,'low'].tail(3).tolist();
    if len(h)>=2 and len(l)>=2:
        if h[-1]>h[-2] and l[-1]>l[-2]: return 'UPTREND'
        if h[-1]<h[-2] and l[-1]<l[-2]: return 'DOWNTREND'
    return 'SIDEWAYS'
def detect_breakout(df,lookback=20):
    o=df.copy(); o['rolling_resistance']=df.high.shift(1).rolling(lookback,min_periods=lookback).max(); o['rolling_support']=df.low.shift(1).rolling(lookback,min_periods=lookback).min(); o['breakout']=o.close>o.rolling_resistance; o['breakdown']=o.close<o.rolling_support; return o
