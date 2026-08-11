from intraday_engine.research.paper_learning import _failure_class


def test_stop_with_weak_volume_is_classified():
    assert _failure_class(
        "STOP",
        '["price is below VWAP"]',
        '["relative volume is weak"]',
    ) == "STOP_WITH_VWAP_CONFLICT"


def test_stop_with_extension_is_classified():
    assert _failure_class(
        "STOP_GAP",
        '[]',
        '["price is extended from VWAP"]',
    ) == "STOP_WHILE_EXTENDED"


def test_timeout_and_win_are_classified():
    assert _failure_class("TIMEOUT", "[]", "[]") == "TIMEOUT"
    assert _failure_class("TARGET", "[]", "[]") == "WIN"
