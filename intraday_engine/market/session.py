"""NSE equity cash-segment session boundaries -- the single source of truth
for "is this timestamp inside a trading session", shared by live-session
orchestration (scripts/paper_session.py) and data-quality/staleness checks
(market/candles.py, research/paper_observer.py). Previously these constants
were only defined inline in scripts/paper_session.py; consolidating them here
means a session-boundary fix can't silently apply to only one caller.

Holiday calendars are not tracked here -- callers already know (or don't
need to know) whether a given date is a trading day; this module only
answers "what time of day is the session", not "is today a trading day".
"""

from __future__ import annotations

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)


def is_session_open(moment: datetime) -> bool:
    """True if `moment` (any tz-aware datetime) falls within 09:15-15:30 IST."""
    local = moment.astimezone(IST)
    return MARKET_OPEN <= local.time() <= MARKET_CLOSE


def is_outside_session(moment: datetime) -> bool:
    return not is_session_open(moment)
