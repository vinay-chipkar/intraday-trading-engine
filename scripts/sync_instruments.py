from intraday_engine.market.universe import sync_instruments
print(sync_instruments().to_string(index=False))
