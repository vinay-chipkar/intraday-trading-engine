from datetime import datetime
from zoneinfo import ZoneInfo

from intraday_engine.market.session import IST, MARKET_CLOSE, MARKET_OPEN, is_outside_session, is_session_open


def test_market_open_boundary_is_inside_session():
    moment = datetime(2026, 8, 13, 9, 15, tzinfo=IST)
    assert is_session_open(moment) is True
    assert is_outside_session(moment) is False


def test_market_close_boundary_is_inside_session():
    moment = datetime(2026, 8, 13, 15, 30, tzinfo=IST)
    assert is_session_open(moment) is True


def test_before_market_open_is_outside_session():
    moment = datetime(2026, 8, 13, 9, 14, 59, tzinfo=IST)
    assert is_session_open(moment) is False
    assert is_outside_session(moment) is True


def test_after_market_close_is_outside_session():
    moment = datetime(2026, 8, 13, 15, 30, 1, tzinfo=IST)
    assert is_session_open(moment) is False


def test_is_session_open_converts_other_timezones():
    utc_moment = datetime(2026, 8, 13, 5, 0, tzinfo=ZoneInfo("UTC"))  # 10:30 IST
    assert is_session_open(utc_moment) is True
