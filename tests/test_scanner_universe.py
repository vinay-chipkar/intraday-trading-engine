import dataclasses

import pandas as pd

import intraday_engine.storage.db as db
from config.settings import settings as real_settings
from intraday_engine.scanner.service import ScannerConfig, scan_universe


class FakeClient:
    def full_market_quotes(self, keys):
        return {
            key: {
                "instrument_token": key,
                "last_price": 100.0,
                "volume": 1000.0,
                "ohlc": {"high": 101.0, "low": 99.0, "close": 100.0},
                "net_change": 1.0,
            }
            for key in keys
        }

    def quote_metrics(self, quotes):
        return {
            key: {
                "ltp": value["last_price"],
                "previous_close": 99.0,
                "change_pct": 1.010101,
            }
            for key, value in quotes.items()
        }


def _patch_scanner(monkeypatch, instruments, liquidity):
    monkeypatch.setattr("intraday_engine.scanner.service._ensure_instrument_master", lambda client: instruments)
    monkeypatch.setattr(
        "intraday_engine.scanner.service._historical_liquidity",
        lambda symbols, trading_date, lookback_days: liquidity,
    )
    monkeypatch.setattr(
        "intraday_engine.scanner.service._current_day_vwap",
        lambda symbols, trading_date: pd.DataFrame({"symbol": symbols, "vwap": [100.0] * len(symbols)}),
    )
    monkeypatch.setattr("intraday_engine.scanner.service.latest_symbol_news_scores", lambda hours: pd.DataFrame())
    monkeypatch.setattr("intraday_engine.scanner.service.latest_market_score", lambda: 0.0)


def test_scan_universe_returns_and_persists_all_ranked_rows(monkeypatch):
    instruments = pd.DataFrame(
        [
            {"symbol": "AAA", "instrument_key": "NSE_EQ|AAA"},
            {"symbol": "BBB", "instrument_key": "NSE_EQ|BBB"},
            {"symbol": "CCC", "instrument_key": "NSE_EQ|CCC"},
        ]
    )
    liquidity = pd.DataFrame([
        {"symbol": s, "avg_daily_volume": 1000.0, "avg_daily_traded_value": 100_000_000.0, "history_days": 20}
        for s in ["AAA", "BBB", "CCC"]
    ])
    _patch_scanner(monkeypatch, instruments, liquidity)
    monkeypatch.setattr(
        "intraday_engine.scanner.service.insert_df",
        lambda table, df: setattr(test_scan_universe_returns_and_persists_all_ranked_rows, "persisted", df),
    )

    ranked = scan_universe(FakeClient(), ScannerConfig(limit=1), trading_date=pd.Timestamp("2026-08-11").date())

    assert [row["symbol"] for row in ranked] == ["AAA", "BBB", "CCC"]
    assert len(test_scan_universe_returns_and_persists_all_ranked_rows.persisted) == 3
    assert set(test_scan_universe_returns_and_persists_all_ranked_rows.persisted["symbol"]) == {"AAA", "BBB", "CCC"}


def test_scanner_universe_config_is_not_limited_by_top_n(monkeypatch):
    instruments = pd.DataFrame(
        [{"symbol": f"S{i:02d}", "instrument_key": f"NSE_EQ|S{i:02d}"} for i in range(12)]
    )
    liquidity = pd.DataFrame([
        {"symbol": f"S{i:02d}", "avg_daily_volume": 1000.0, "avg_daily_traded_value": 100_000_000.0, "history_days": 20}
        for i in range(12)
    ])
    _patch_scanner(monkeypatch, instruments, liquidity)
    monkeypatch.setattr(
        "intraday_engine.scanner.service.insert_df",
        lambda table, df: setattr(test_scanner_universe_config_is_not_limited_by_top_n, "persisted", df),
    )

    ranked = scan_universe(FakeClient(), ScannerConfig(limit=5), trading_date=pd.Timestamp("2026-08-11").date())

    assert len(ranked) == 12
    assert len(test_scanner_universe_config_is_not_limited_by_top_n.persisted) == 12


def test_scanner_keeps_quoted_stocks_without_historical_liquidity(monkeypatch):
    instruments = pd.DataFrame(
        [{"symbol": f"S{i:02d}", "instrument_key": f"NSE_EQ|S{i:02d}"} for i in range(8)]
    )
    liquidity = pd.DataFrame([
        {"symbol": "S00", "avg_daily_volume": 1000.0, "avg_daily_traded_value": 100_000_000.0, "history_days": 20},
    ])
    _patch_scanner(monkeypatch, instruments, liquidity)
    monkeypatch.setattr(
        "intraday_engine.scanner.service.insert_df",
        lambda table, df: setattr(test_scanner_keeps_quoted_stocks_without_historical_liquidity, "persisted", df),
    )

    ranked = scan_universe(FakeClient(), ScannerConfig(limit=5), trading_date=pd.Timestamp("2026-08-11").date())

    assert len(ranked) == 8
    assert len(test_scanner_keeps_quoted_stocks_without_historical_liquidity.persisted) == 8
    assert {row["symbol"] for row in ranked} == {f"S{i:02d}" for i in range(8)}
    incomplete = next(row for row in ranked if row["symbol"] == "S01")
    assert incomplete["history_days"] == 0
    assert incomplete["data_quality"] == "MISSING_HISTORY"
    assert incomplete["rvol_valid"] is False
    assert incomplete["liquidity_valid"] is False
    assert "DATA_INCOMPLETE" in incomplete["reason"]


def test_incomplete_history_cannot_outrank_complete_history(monkeypatch):
    instruments = pd.DataFrame([
        {"symbol": "COMPLETE", "instrument_key": "NSE_EQ|COMPLETE"},
        {"symbol": "MISSING", "instrument_key": "NSE_EQ|MISSING"},
    ])
    liquidity = pd.DataFrame([
        {"symbol": "COMPLETE", "avg_daily_volume": 1000.0, "avg_daily_traded_value": 100_000_000.0, "history_days": 20},
    ])
    _patch_scanner(monkeypatch, instruments, liquidity)
    monkeypatch.setattr("intraday_engine.scanner.service.insert_df", lambda table, df: None)

    ranked = scan_universe(FakeClient(), ScannerConfig(limit=1), trading_date=pd.Timestamp("2026-08-11").date())

    assert ranked[0]["symbol"] == "COMPLETE"
    missing = next(row for row in scan_universe(FakeClient(), ScannerConfig(limit=1), trading_date=pd.Timestamp("2026-08-11").date()) if row["symbol"] == "MISSING")
    assert missing["candidate_score"] < 10.0


def test_scan_universe_persists_columns_in_the_exact_candidate_events_schema_order(monkeypatch, tmp_path):
    # storage/db.py::insert_df runs `INSERT INTO table SELECT * FROM incoming_df`
    # -- a POSITIONAL insert, not by column name. candidate_events is built from
    # a 14-column CREATE TABLE plus 4 columns added later via ALTER TABLE
    # (history_days, data_quality, rvol_valid, liquidity_valid), so the only way
    # to know the true runtime column order is to ask the schema, not to read
    # the CREATE TABLE statement by eye. A service.py that persists a DataFrame
    # with the wrong column count/order would either raise (count mismatch) or,
    # worse, silently misalign values into the wrong columns.
    fake_settings = dataclasses.replace(real_settings, duckdb_path=str(tmp_path / "schema_check.duckdb"))
    monkeypatch.setattr(db, "settings", fake_settings)
    connection = db.conn()
    try:
        schema_columns = [
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'candidate_events' ORDER BY ordinal_position"
            ).fetchall()
        ]
    finally:
        connection.close()

    instruments = pd.DataFrame([{"symbol": "AAA", "instrument_key": "NSE_EQ|AAA"}])
    liquidity = pd.DataFrame([
        {"symbol": "AAA", "avg_daily_volume": 1000.0, "avg_daily_traded_value": 100_000_000.0, "history_days": 20},
    ])
    _patch_scanner(monkeypatch, instruments, liquidity)
    persisted = {}
    monkeypatch.setattr(
        "intraday_engine.scanner.service.insert_df",
        lambda table, df: persisted.setdefault("df", df),
    )

    scan_universe(FakeClient(), ScannerConfig(limit=1), trading_date=pd.Timestamp("2026-08-11").date())

    assert list(persisted["df"].columns) == schema_columns
