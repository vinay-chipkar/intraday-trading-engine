from __future__ import annotations

from datetime import date
from urllib.parse import quote
import os

import pandas as pd
import requests


BASE = "https://api.upstox.com"


class UpstoxREST:
    """Small REST boundary around the Upstox endpoints used by the research engine."""

    def __init__(self, access_token: str | None = None, timeout: int = 20):
        self.access_token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not self.access_token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is not set")
        self.session = requests.Session()
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = self.session.get(
            BASE + path,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search_instruments(
        self,
        query: str,
        exchanges: str = "NSE",
        segments: str = "EQ",
        records: int = 30,
        page_number: int = 1,
    ) -> list[dict]:
        payload = self._get(
            "/v2/instruments/search",
            {
                "query": query,
                "exchanges": exchanges,
                "segments": segments,
                "page_number": page_number,
                "records": min(records, 30),
            },
        )
        return payload.get("data", [])

    def resolve_equity(self, symbol: str) -> dict:
        symbol = symbol.upper().strip()
        for row in self.search_instruments(symbol):
            if row.get("trading_symbol", "").upper() == symbol and row.get("instrument_key"):
                return row
        raise LookupError(f"NSE equity instrument not found: {symbol}")

    @staticmethod
    def _frame(candles: list[list]) -> pd.DataFrame:
        columns = [
            "timestamp", "open", "high", "low", "close", "volume", "open_interest"
        ]
        frame = pd.DataFrame(candles, columns=columns)
        if frame.empty:
            return frame

        frame["timestamp"] = (
            pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            .dt.tz_convert("Asia/Kolkata")
        )
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        return (
            frame.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )

    def intraday_candles(
        self, instrument_key: str, unit: str = "minutes", interval: int = 1
    ) -> pd.DataFrame:
        key = quote(instrument_key, safe="")
        payload = self._get(
            f"/v3/historical-candle/intraday/{key}/{unit}/{interval}"
        )
        return self._frame(payload.get("data", {}).get("candles", []))

    def historical_candles(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        to_date: date,
        from_date: date | None = None,
    ) -> pd.DataFrame:
        key = quote(instrument_key, safe="")
        path = f"/v3/historical-candle/{key}/{unit}/{interval}/{to_date.isoformat()}"
        if from_date:
            path += f"/{from_date.isoformat()}"
        payload = self._get(path)
        return self._frame(payload.get("data", {}).get("candles", []))

    def full_market_quotes(self, instrument_keys: list[str]) -> dict:
        if not instrument_keys:
            return {}
        payload = self._get(
            "/v2/market-quote/quotes",
            {"instrument_key": ",".join(instrument_keys)},
        )
        return payload.get("data", {})
