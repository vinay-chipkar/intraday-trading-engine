import pandas as pd

from scripts.backfill_history import backfill


class FakeClient:
    def historical_candles(self, instrument_key, unit, interval, to_date, from_date):
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-08-01T09:15:00+05:30"]),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
                "open_interest": [0.0],
            }
        )


def test_backfill_uses_full_instrument_master(monkeypatch, capsys):
    instruments = pd.DataFrame(
        [
            {"symbol": "AAA", "instrument_key": "NSE_EQ|AAA"},
            {"symbol": "BBB", "instrument_key": "NSE_EQ|BBB"},
        ]
    )
    inserted = []

    monkeypatch.setattr("scripts.backfill_history.UpstoxREST", FakeClient)
    monkeypatch.setattr("scripts.backfill_history.get_instruments", lambda: instruments)
    monkeypatch.setattr("scripts.backfill_history.insert_candles", lambda frame: inserted.append(frame) or 1)

    result = backfill(days=30)

    assert result == 2
    assert len(inserted) == 2
    assert {frame.iloc[0]["symbol"] for frame in inserted} == {"AAA", "BBB"}
    assert "Universe: 2 symbols" in capsys.readouterr().out
