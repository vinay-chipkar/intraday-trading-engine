import pandas as pd

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


def test_scan_universe_returns_and_persists_all_ranked_rows(monkeypatch):
    instruments = pd.DataFrame(
        [
            {"symbol": "AAA", "instrument_key": "NSE_EQ|AAA"},
            {"symbol": "BBB", "instrument_key": "NSE_EQ|BBB"},
            {"symbol": "CCC", "instrument_key": "NSE_EQ|CCC"},
        ]
    )

    monkeypatch.setattr(
        "intraday_engine.scanner.service._ensure_instrument_master",
        lambda client: instruments,
    )
    monkeypatch.setattr(
        "intraday_engine.scanner.service._historical_liquidity",
        lambda symbols, trading_date, lookback_days: pd.DataFrame(
            [
                {"symbol": s, "avg_daily_volume": 1000.0, "avg_daily_traded_value": 100_000_000.0, "history_days": 20}
                for s in symbols
            ]
        ),
    )
    monkeypatch.setattr(
        "intraday_engine.scanner.service._current_day_vwap",
        lambda symbols, trading_date: pd.DataFrame({"symbol": symbols, "vwap": [100.0] * len(symbols)}),
    )
    monkeypatch.setattr(
        "intraday_engine.scanner.service.latest_symbol_news_scores",
        lambda hours: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "intraday_engine.scanner.service.latest_market_score",
        lambda: 0.0,
    )
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
    monkeypatch.setattr("intraday_engine.scanner.service._ensure_instrument_master", lambda client: instruments)
    monkeypatch.setattr(
        "intraday_engine.scanner.service._historical_liquidity",
        lambda symbols, trading_date, lookback_days: pd.DataFrame(
            [{"symbol": s, "avg_daily_volume": 1000.0, "avg_daily_traded_value": 100_000_000.0, "history_days": 20} for s in symbols]
        ),
    )
    monkeypatch.setattr(
        "intraday_engine.scanner.service._current_day_vwap",
        lambda symbols, trading_date: pd.DataFrame({"symbol": symbols, "vwap": [100.0] * len(symbols)}),
    )
    monkeypatch.setattr("intraday_engine.scanner.service.latest_symbol_news_scores", lambda hours: pd.DataFrame())
    monkeypatch.setattr("intraday_engine.scanner.service.latest_market_score", lambda: 0.0)
    monkeypatch.setattr("intraday_engine.scanner.service.insert_df", lambda table, df: setattr(test_scanner_universe_config_is_not_limited_by_top_n, "persisted", df))

    ranked = scan_universe(FakeClient(), ScannerConfig(limit=5), trading_date=pd.Timestamp("2026-08-11").date())

    assert len(ranked) == 12
    assert len(test_scanner_universe_config_is_not_limited_by_top_n.persisted) == 12
