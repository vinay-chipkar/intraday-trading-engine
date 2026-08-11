import pytest

from intraday_engine.research.daily_cycle import run_daily_cycle


def test_daily_cycle_refuses_live_mode():
    with pytest.raises(RuntimeError, match="Only PAPER mode"):
        run_daily_cycle(mode="LIVE")
