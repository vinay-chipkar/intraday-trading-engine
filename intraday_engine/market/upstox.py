from __future__ import annotations
from datetime import date
from urllib.parse import quote
import os
import pandas as pd
import requests

BASE = "https://api.upstox.com"

class UpstoxREST:
    def __init__(self, access_token: str | None = None, timeout: int = 20):
        self.access_token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not self.access_token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is not set")
        self.session = requests.Session(); self.timeout = timeout

    @property
    def headers(self):
        return {"Accept": "application/json", "Authorization": f"Bearer {self.access_token}"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.session.get(BASE + path, headers=self.headers, params=params, timeout=self.timeout)
        r.raise_for_status(); return r.json()

    def search_instruments(self, query: str, exchanges="NSE", segments="EQ", records=30):
        return self._get("/v2/instruments/search", {"query":query,"exchanges":exchanges,"segments":segments,"page_number":1,"records":min(records,30)}).get("data", [])

    def resolve_equity(self, symbol: str) -> dict:
        for row in self.search_instruments(symbol):
            if row.get("trading_symbol", "").upper() == symbol.upper() and row.get("instrument_key"):
                return row
        raise LookupError(f"NSE equity instrument not found: {symbol}")

    @staticmethod
    def _frame(candles):
        cols=["timestamp","open","high","low","close","volume","open_interest"]
        df=pd.DataFrame(candles, columns=cols)
        if df.empty: return df
        df["timestamp"]=pd.to_datetime(df.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        for c in cols[1:]: df[c]=pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def intraday_candles(self, instrument_key: str, unit="minutes", interval=1):
        key=quote(instrument_key, safe="")
        return self._frame(self._get(f"/v3/historical-candle/intraday/{key}/{unit}/{interval}").get("data",{}).get("candles",[]))

    def historical_candles(self, instrument_key: str, unit: str, interval: int, to_date: date, from_date: date|None=None):
        key=quote(instrument_key, safe="")
        path=f"/v3/historical-candle/{key}/{unit}/{interval}/{to_date.isoformat()}"
        if from_date: path += f"/{from_date.isoformat()}"
        return self._frame(self._get(path).get("data",{}).get("candles",[]))
