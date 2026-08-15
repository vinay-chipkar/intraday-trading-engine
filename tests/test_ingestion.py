import pytest

from intraday_engine.market.ingestion import IngestionFailure, IngestionResult, assess_ingestion_results


def _result(symbol: str, *, error: str | None = None) -> IngestionResult:
    return IngestionResult(
        symbol=symbol,
        rows_received=0 if error else 5,
        rows_inserted=0 if error else 5,
        last_timestamp=None,
        quality={},
        error=error,
    )


def test_raises_when_all_symbols_fail():
    results = [_result("A", error="401 Unauthorized"), _result("B", error="401 Unauthorized")]
    with pytest.raises(IngestionFailure, match="2/2 symbols"):
        assess_ingestion_results(results)


def test_tolerates_a_single_failed_symbol_out_of_many():
    results = [_result(s) for s in "ABCDE"] + [_result("F", error="timeout")]
    assess_ingestion_results(results)  # must not raise


def test_raises_when_majority_of_symbols_fail():
    # 3/5 failed (60%) is not "one or a few" -- below the 50% success floor.
    results = [_result("A"), _result("B")] + [_result(s, error="boom") for s in "CDE"]
    with pytest.raises(IngestionFailure, match="3/5 symbols"):
        assess_ingestion_results(results)


def test_exactly_half_success_is_tolerated():
    results = [_result("A"), _result("B")] + [_result(s, error="boom") for s in "CD"]
    assess_ingestion_results(results)  # 50% success == the floor, not below it


def test_raises_on_empty_results():
    with pytest.raises(IngestionFailure, match="no results"):
        assess_ingestion_results([])


def _empty_response_result(symbol: str) -> IngestionResult:
    # A well-formed response with zero candles and no exception -- must not
    # be treated as a healthy symbol.
    return IngestionResult(
        symbol=symbol, rows_received=0, rows_inserted=0, last_timestamp=None, quality={}, error=None,
    )


def test_empty_response_with_no_error_is_not_healthy():
    results = [_empty_response_result(s) for s in "ABCDE"] + [_result("F")]
    with pytest.raises(IngestionFailure, match="5/6 symbols"):
        assess_ingestion_results(results)


def test_empty_response_failure_message_distinguishes_from_a_real_error():
    results = [_empty_response_result("A"), _result("B", error="timeout")]
    with pytest.raises(IngestionFailure, match=r"A: empty response.*B: timeout"):
        assess_ingestion_results(results)
