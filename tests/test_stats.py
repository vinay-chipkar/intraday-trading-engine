from intraday_engine.research.stats import candidate_score_bucket, rsi_bucket


def test_candidate_score_bucket_boundaries():
    assert candidate_score_bucket(10.0) == "<30"
    assert candidate_score_bucket(30.0) == "30-49"
    assert candidate_score_bucket(49.9) == "30-49"
    assert candidate_score_bucket(50.0) == "50-69"
    assert candidate_score_bucket(69.9) == "50-69"
    assert candidate_score_bucket(70.0) == "70+"
    assert candidate_score_bucket(float("nan")) == "unknown"


def test_rsi_bucket_boundaries():
    assert rsi_bucket(10.0) == "<30 (oversold)"
    assert rsi_bucket(35.0) == "30-48 (bearish)"
    assert rsi_bucket(50.0) == "48-52 (neutral)"
    assert rsi_bucket(60.0) == "52-70 (bullish)"
    assert rsi_bucket(85.0) == ">70 (overbought)"
    assert rsi_bucket(float("nan")) == "unknown"
