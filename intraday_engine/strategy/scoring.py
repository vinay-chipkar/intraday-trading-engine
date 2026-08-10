def score_long(r,market_score=0):
    s=0; reasons=[]
    if r.get('relative_volume',0)>=2:s+=20;reasons.append('RVOL>=2x')
    elif r.get('relative_volume',0)>=1.5:s+=10;reasons.append('RVOL>=1.5x')
    if r.get('above_vwap'):s+=10;reasons.append('ABOVE_VWAP')
    if r.get('trend')=='UPTREND':s+=15;reasons.append('UPTREND')
    if r.get('breakout'):s+=20;reasons.append('BREAKOUT')
    if 55<=r.get('rsi14',50)<=70:s+=10;reasons.append('RSI_55_70')
    if r.get('candle_pattern') in {'HAMMER','BULLISH_ENGULFING'}:s+=8;reasons.append('BULL_CANDLE')
    return min(100,s+max(-10,min(10,market_score*.2))),reasons
def score_short(r,market_score=0):
    s=0; reasons=[]
    if r.get('relative_volume',0)>=2:s+=20;reasons.append('RVOL>=2x')
    elif r.get('relative_volume',0)>=1.5:s+=10;reasons.append('RVOL>=1.5x')
    if not r.get('above_vwap'):s+=10;reasons.append('BELOW_VWAP')
    if r.get('trend')=='DOWNTREND':s+=15;reasons.append('DOWNTREND')
    if r.get('breakdown'):s+=20;reasons.append('BREAKDOWN')
    if 30<=r.get('rsi14',50)<=45:s+=10;reasons.append('RSI_30_45')
    if r.get('candle_pattern') in {'SHOOTING_STAR','BEARISH_ENGULFING'}:s+=8;reasons.append('BEAR_CANDLE')
    return min(100,s+max(-10,min(10,-market_score*.2))),reasons
