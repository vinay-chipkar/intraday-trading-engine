from __future__ import annotations

import logging
import time
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

# 429 (rate limited) and 5xx (server-side) are worth a bounded retry; any other
# 4xx (bad request, unauthorized, not found) will not resolve itself on retry.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

LOGGER = logging.getLogger(__name__)


class UpstoxAPIError(RuntimeError):
    """Raised when an Upstox response is not the JSON object shape every
    caller in this module assumes -- a malformed/unexpected body must fail
    loudly here rather than let a caller's blind .get("data", {}) silently
    produce an empty-looking (but not explicitly flagged) result."""


class AmbiguousInstrumentError(RuntimeError):
    """Raised when instrument search returns more than one distinct
    instrument_key for what should be a single, unambiguous trading symbol
    (e.g. after a corporate action/relisting) -- resolving to "whichever the
    API listed first" would be a silent, undetectable wrong-security risk."""


class UpstoxREST:
    def __init__(
        self,
        access_token: str | None = None,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
    ):
        self.access_token = access_token or settings.upstox_access_token
        if not self.access_token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is not set")
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def _sleep_before_retry(self, attempt: int, cause: object, path: str) -> None:
        wait = self.backoff_base_seconds * (2 ** attempt)
        LOGGER.warning(
            "Upstox request to %s failed (%s); retrying in %.1fs (attempt %d/%d)",
            path, cause, wait, attempt + 1, self.max_retries,
        )
        time.sleep(wait)

    def _get(self, path: str, params: dict | None = None) -> dict:
        """GET path, retrying transient failures with bounded exponential
        backoff, and guaranteeing the return value is a JSON object.

        Retried: connection errors, timeouts, HTTP 429, HTTP 5xx.
        Not retried: any other 4xx (won't resolve itself), malformed/non-JSON
        bodies (raised immediately as UpstoxAPIError).
        """
        attempt = 0
        while True:
            try:
                response = self.session.get(
                    BASE + path, headers=self.headers, params=params, timeout=self.timeout,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise
                self._sleep_before_retry(attempt, exc, path)
                attempt += 1
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                self._sleep_before_retry(attempt, response.status_code, path)
                attempt += 1
                continue

            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise UpstoxAPIError(f"malformed (non-JSON) response body from {path}") from exc
            if not isinstance(payload, dict):
                raise UpstoxAPIError(
                    f"unexpected response shape from {path}: expected a JSON object, "
                    f"got {type(payload).__name__}"
                )
            return payload

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
        data = payload.get("data") or []
        return data if isinstance(data, list) else []

    def resolve_equity(self, symbol: str) -> dict:
        symbol = symbol.upper().strip()
        candidates = [symbol]
        alias = SYMBOL_ALIASES.get(symbol)
        if alias and alias not in candidates:
            candidates.append(alias)

        for candidate in candidates:
            matches = [
                row
                for row in self.search_instruments(candidate)
                if str(row.get("trading_symbol", "")).upper() == candidate and row.get("instrument_key")
            ]
            if not matches:
                continue
            distinct_keys = {row["instrument_key"] for row in matches}
            if len(distinct_keys) > 1:
                # Same trading_symbol resolving to more than one instrument_key
                # (e.g. mid-relisting) is exactly the silent wrong-security risk
                # this function must never guess through -- fail loudly instead
                # of returning "whichever the API happened to list first".
                raise AmbiguousInstrumentError(
                    f"{symbol} resolved to {len(distinct_keys)} distinct instrument_keys "
                    f"via candidate {candidate!r}: {sorted(distinct_keys)}"
                )
            return matches[0]

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

    @staticmethod
    def _data_dict(payload: dict) -> dict:
        """payload["data"] is expected to be an object; treat an explicit
        null the same as a missing key rather than propagating None into a
        caller's .get(...) chain."""
        data = payload.get("data")
        return data if isinstance(data, dict) else {}

    def intraday_candles(
        self, instrument_key: str, unit: str = "minutes", interval: int = 1
    ) -> pd.DataFrame:
        key = quote(instrument_key, safe="")
        payload = self._get(
            f"/v3/historical-candle/intraday/{key}/{unit}/{interval}"
        )
        candles = self._data_dict(payload).get("candles") or []
        return self._frame(candles)

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
        candles = self._data_dict(payload).get("candles") or []
        return self._frame(candles)

    def full_market_quotes(self, instrument_keys: list[str]) -> dict:
        if not instrument_keys:
            return {}
        payload = self._get(
            "/v2/market-quote/quotes",
            {"instrument_key": ",".join(instrument_keys)},
        )
        return self._data_dict(payload)

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
        return self._data_dict(payload)

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
