from __future__ import annotations

from datetime import date
from urllib.parse import quote

import pandas as pd
import requests

from config.settings import settings

BASE = "https://api.upstox.com"

SYMBOL_ALIASES = {
    "BAJAJ_AUTO": "BAJAJ-AUTO",
    "ZOMATO": "ETERNAL",
    "TATAMOTORS": "TMPV",
}


class UpstoxREST:
    def __init__(self, access_token: str | None = None, timeout: int = 15):
        self.access_token = access_token or settings.upstox_access_token
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
        candidates = [symbol]
        alias = SYMBOL_ALIASES.get(symbol)
        if alias and alias not in candidates:
            candidates.append(alias)

        for candidate in candidates:
            for row in self.search_instruments(candidate):
                trading_symbol = str(row.get("trading_symbol", "")).upper()
                if trading_symbol == candidate and row.get("instrument_key"):
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

    def news(self, instrument_keys: list[str], page_size: int = 100) -> dict:
        """Return recent Upstox news grouped by instrument key."""
        if not instrument_keys:
            return {}
        if len(instrument_keys) > 30:
            raise ValueError("Upstox News API accepts at most 30 instrument keys per request")
        payload = self._get(
            "/v2/news",
            {
                "category": "instrument_keys",
                "instrument_keys": ",".join(instrument_keys),
                "page_number": 1,
                "page_size": min(page_size, 100),
            },
        )
        return payload.get("data", {})

    @staticmethod
    def quote_metrics(quotes: dict) -> dict[str, dict[str, float | None]]:
        """Normalize full quotes using Upstox net_change when available."""
        normalized: dict[str, dict[str, float | None]] = {}
        for key, raw in quotes.items():
            if not isinstance(raw, dict):
                continue

            ltp_raw = raw.get("last_price", raw.get("ltp"))
            net_change_raw = raw.get("net_change")
            prev_close_raw = raw.get("prev_close")

            try:
                ltp = float(ltp_raw) if ltp_raw is not None else None
            except (TypeError, ValueError):
                ltp = None

            try:
                net_change = float(net_change_raw) if net_change_raw is not None else None
            except (TypeError, ValueError):
                net_change = None

            try:
                previous_close = float(prev_close_raw) if prev_close_raw is not None else None
            except (TypeError, ValueError):
                previous_close = None

            if previous_close is None and ltp is not None and net_change is not None:
                previous_close = ltp - net_change

            change_pct = None
            if previous_close not in (None, 0) and ltp is not None:
                change_pct = (ltp / previous_close - 1.0) * 100.0

            normalized[key] = {
                "ltp": ltp,
                "previous_close": previous_close,
                "net_change": net_change,
                "change_pct": change_pct,
            }
        return normalized
