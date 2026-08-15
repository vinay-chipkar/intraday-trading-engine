"""Coverage for the single network boundary module (intraday_engine/market/upstox.py).

Before this, no test exercised UpstoxREST's own HTTP handling at all --
retries, malformed responses, and instrument-resolution ambiguity were
entirely unverified. These tests never hit the network: session.get is
replaced with a fake that returns a scripted sequence of responses.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from intraday_engine.market.upstox import (
    AmbiguousInstrumentError,
    UpstoxAPIError,
    UpstoxREST,
)


def _client(monkeypatch, **kwargs) -> UpstoxREST:
    client = UpstoxREST(access_token="test-token", **kwargs)
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # no real delay in tests
    return client


def _response(status_code: int = 200, json_body=None, json_error: bool = False):
    resp = Mock()
    resp.status_code = status_code
    if json_error:
        resp.json.side_effect = ValueError("invalid JSON")
    else:
        resp.json.return_value = json_body
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_retries_on_connection_error_then_succeeds(monkeypatch):
    client = _client(monkeypatch)
    ok = _response(json_body={"data": {"candles": []}})
    client.session.get = Mock(side_effect=[requests.exceptions.ConnectionError("boom"), ok])
    payload = client._get("/some/path")
    assert payload == {"data": {"candles": []}}
    assert client.session.get.call_count == 2


def test_retries_on_429_then_succeeds(monkeypatch):
    client = _client(monkeypatch)
    rate_limited = _response(status_code=429)
    ok = _response(json_body={"data": {}})
    client.session.get = Mock(side_effect=[rate_limited, ok])
    payload = client._get("/some/path")
    assert payload == {"data": {}}
    assert client.session.get.call_count == 2


def test_retries_on_5xx_then_succeeds(monkeypatch):
    client = _client(monkeypatch)
    server_error = _response(status_code=503)
    ok = _response(json_body={"data": {}})
    client.session.get = Mock(side_effect=[server_error, ok])
    client._get("/some/path")
    assert client.session.get.call_count == 2


def test_does_not_retry_on_404(monkeypatch):
    client = _client(monkeypatch)
    not_found = _response(status_code=404)
    client.session.get = Mock(return_value=not_found)
    with pytest.raises(requests.exceptions.HTTPError):
        client._get("/some/path")
    assert client.session.get.call_count == 1  # never retried


def test_raises_after_exhausting_retries(monkeypatch):
    client = _client(monkeypatch, max_retries=2)
    client.session.get = Mock(side_effect=requests.exceptions.Timeout("slow"))
    with pytest.raises(requests.exceptions.Timeout):
        client._get("/some/path")
    assert client.session.get.call_count == 3  # initial + 2 retries


def test_malformed_non_json_response_raises_upstox_api_error(monkeypatch):
    client = _client(monkeypatch)
    client.session.get = Mock(return_value=_response(json_error=True))
    with pytest.raises(UpstoxAPIError, match="malformed"):
        client._get("/some/path")


def test_non_dict_json_response_raises_upstox_api_error(monkeypatch):
    client = _client(monkeypatch)
    client.session.get = Mock(return_value=_response(json_body=["not", "a", "dict"]))
    with pytest.raises(UpstoxAPIError, match="expected a JSON object"):
        client._get("/some/path")


def test_intraday_candles_tolerates_null_data_and_null_candles(monkeypatch):
    client = _client(monkeypatch)
    client.session.get = Mock(return_value=_response(json_body={"data": {"candles": None}}))
    frame = client.intraday_candles("NSE_EQ|TEST")
    assert frame.empty

    client.session.get = Mock(return_value=_response(json_body={"data": None}))
    frame2 = client.intraday_candles("NSE_EQ|TEST")
    assert frame2.empty


def test_resolve_equity_returns_the_single_unambiguous_match(monkeypatch):
    client = _client(monkeypatch)
    client.session.get = Mock(
        return_value=_response(
            json_body={"data": [{"trading_symbol": "INFY", "instrument_key": "NSE_EQ|INFY1"}]}
        )
    )
    row = client.resolve_equity("infy")
    assert row["instrument_key"] == "NSE_EQ|INFY1"


def test_resolve_equity_raises_on_ambiguous_distinct_instrument_keys(monkeypatch):
    client = _client(monkeypatch)
    client.session.get = Mock(
        return_value=_response(
            json_body={
                "data": [
                    {"trading_symbol": "INFY", "instrument_key": "NSE_EQ|OLD_INFY"},
                    {"trading_symbol": "INFY", "instrument_key": "NSE_EQ|NEW_INFY"},
                ]
            }
        )
    )
    with pytest.raises(AmbiguousInstrumentError, match="INFY"):
        client.resolve_equity("INFY")


def test_resolve_equity_tolerates_harmless_duplicate_rows(monkeypatch):
    # Same instrument_key appearing twice (e.g. paginated/duplicated search
    # results) is not ambiguous -- it's the same security both times.
    client = _client(monkeypatch)
    client.session.get = Mock(
        return_value=_response(
            json_body={
                "data": [
                    {"trading_symbol": "INFY", "instrument_key": "NSE_EQ|INFY1"},
                    {"trading_symbol": "INFY", "instrument_key": "NSE_EQ|INFY1"},
                ]
            }
        )
    )
    row = client.resolve_equity("INFY")
    assert row["instrument_key"] == "NSE_EQ|INFY1"


def test_resolve_equity_falls_back_to_known_alias(monkeypatch):
    client = _client(monkeypatch)
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params["query"])
        if params["query"] == "ZOMATO":
            return _response(json_body={"data": []})
        return _response(
            json_body={"data": [{"trading_symbol": "ETERNAL", "instrument_key": "NSE_EQ|ETERNAL1"}]}
        )

    client.session.get = Mock(side_effect=fake_get)
    row = client.resolve_equity("ZOMATO")
    assert row["instrument_key"] == "NSE_EQ|ETERNAL1"
    assert calls == ["ZOMATO", "ETERNAL"]


def test_resolve_equity_raises_lookup_error_when_nothing_matches(monkeypatch):
    client = _client(monkeypatch)
    client.session.get = Mock(return_value=_response(json_body={"data": []}))
    with pytest.raises(LookupError, match="NOTASYMBOL"):
        client.resolve_equity("NOTASYMBOL")
