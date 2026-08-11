from datetime import datetime

import pytest

from intraday_engine.market.context import (
    build_context,
    classify,
    score_market,
    values_from_quotes,
    vix_penalty,
)
from intraday_engine.market.upstox import UpstoxREST


def test_classify_regimes():
    assert classify(35) == "BULLISH"
    assert classify(15) == "MILD_BULLISH"
    assert classify(0) == "NEUTRAL"
    assert classify(-15) == "MILD_BEARISH"
    assert classify(-35) == "BEARISH"


def test_score_is_deterministic():
    values = {
        "gift_nifty_change_pct": 1.0,
        "dow_change_pct": 0.5,
        "sp500_change_pct": 0.5,
        "nasdaq_change_pct": 1.0,
        "nifty_change_pct": 0.8,
        "banknifty_change_pct": 0.4,
        "india_vix_penalty": 0.0,
    }
    assert score_market(values) == pytest.approx(11.7)


def test_vix_penalty_only_applies_above_baseline():
    assert vix_penalty(14) == 0
    assert vix_penalty(15) == 0
    assert vix_penalty(20) == 3.75


def test_quote_metrics_and_context_mapping():
    keys = {
        "GLOBAL_INDEX|SGX NIFTY": {"last_price": 101.0, "net_change": 1.0},
        "NSE_INDEX|Nifty 50": {"last_price": 202.0, "net_change": 2.0},
        "NSE_INDEX|Nifty Bank": {"last_price": 99.0, "net_change": -1.0},
        "NSE_INDEX|India VIX": {"last_price": 18.0},
    }
    metrics = UpstoxREST.quote_metrics(keys)
    values = values_from_quotes(metrics)
    assert round(values["gift_nifty_change_pct"], 4) == round(1.0 / 100.0 * 100, 4)
    assert round(values["nifty_change_pct"], 4) == round(2.0 / 200.0 * 100, 4)
    assert round(values["banknifty_change_pct"], 4) == round(-1.0 / 100.0 * 100, 4)
    assert values["india_vix"] == 18.0


def test_quote_metrics_fallback_to_explicit_prev_close():
    metrics = UpstoxREST.quote_metrics(
        {"NSE_EQ|TEST": {"last_price": 105.0, "prev_close": 100.0}}
    )
    assert metrics["NSE_EQ|TEST"]["previous_close"] == 100.0
    assert metrics["NSE_EQ|TEST"]["change_pct"] == pytest.approx(5.0)


def test_build_context_has_timestamp_and_regime():
    context = build_context(
        {
            "gift_nifty_change_pct": 2.0,
            "nifty_change_pct": 1.5,
            "banknifty_change_pct": 1.0,
            "india_vix": 14.0,
            "india_vix_penalty": 0.0,
        }
    )
    assert isinstance(context.captured_at, datetime)
    assert context.regime == "MILD_BULLISH"
