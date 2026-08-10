from pathlib import Path
import pandas as pd
from config.settings import settings
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.storage.db import upsert_instruments
DEFAULT_PATH=Path(__file__).resolve().parents[2]/'config'/'universe.csv'
def load_symbols(path=DEFAULT_PATH):return pd.read_csv(path)['symbol'].dropna().astype(str).str.upper().tolist()
def sync_instruments(symbols=None):
    api=UpstoxREST(settings.upstox_access_token); rows=[]
    for s in symbols or load_symbols():
        try:
            x=api.resolve_equity(s); rows.append({'symbol':s,'instrument_key':x['instrument_key'],'name':x.get('name'),'trading_symbol':x.get('trading_symbol'),'updated_at':pd.Timestamp.now(tz='Asia/Kolkata')})
        except Exception as e: print(f'WARN {s}: {e}')
    df=pd.DataFrame(rows); upsert_instruments(df); return df
