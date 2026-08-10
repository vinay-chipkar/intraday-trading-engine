import pandas as pd

def classify_candle(r):
    o,h,l,c=map(float,(r.open,r.high,r.low,r.close)); body=abs(c-o); rng=max(h-l,1e-9); upper=h-max(o,c); lower=min(o,c)-l
    if body/rng<=.1:return 'DOJI'
    if lower>=2*body and upper<=body*.6:return 'HAMMER' if c>=o else 'HANGING_MAN'
    if upper>=2*body and lower<=body*.6:return 'INVERTED_HAMMER' if c>=o else 'SHOOTING_STAR'
    return 'BULLISH' if c>o else 'BEARISH'
def add_candle_patterns(df):
    o=df.copy(); o['candle_pattern']=o.apply(classify_candle,axis=1); po,pc=o.open.shift(1),o.close.shift(1); be=(pc<po)&(o.close>o.open)&(o.open<=pc)&(o.close>=po); se=(pc>po)&(o.close<o.open)&(o.open>=pc)&(o.close<=po); o.loc[be,'candle_pattern']='BULLISH_ENGULFING'; o.loc[se,'candle_pattern']='BEARISH_ENGULFING'; return o
