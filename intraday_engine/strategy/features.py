from intraday_engine.technical.indicators import add_indicators
from intraday_engine.technical.structure import support_resistance,trend_from_swings,detect_breakout
from intraday_engine.patterns.candles import add_candle_patterns
def enrich(df):
    o=add_indicators(df); o=add_candle_patterns(o); o=detect_breakout(o); s,r=support_resistance(o); o['support']=s; o['resistance']=r; o['trend']=trend_from_swings(o); o['distance_to_support_pct']=(o.close-o.support)/o.close*100; o['distance_to_resistance_pct']=(o.resistance-o.close)/o.close*100; o['above_vwap']=o.close>o.vwap; return o
