from intraday_engine.paper.simulator import PaperBroker

def test_position_sizing():
    assert PaperBroker(100000,.005).size(100,98)==250
