import pandas as pd
from intraday_engine.patterns.candles import classify_candle

def test_hammer():
    row=pd.Series({'open':100,'high':101,'low':95,'close':100.8}); assert classify_candle(row)=='HAMMER'
