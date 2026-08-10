from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd

from intraday_engine.technical.indicators import add_indicators


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def technical_feature_score(row: pd.Series) -> float:
    """Transparent feature-quality score; this is not a trade signal."""
    score = 50.0
    if row.get("trend") == "UPTREND":
        score += 10.0
    elif row.get("trend") == "DOWNTREND":
        score -= 10.0

    rsi = float(row.get("rsi14") or 50.0)
    if 50.0 <= rsi <= 70.0:
        score += 5.0
    elif 30.0 <= rsi < 50.0:
        score -= 5.0

    adx_value = float(row.get("adx14") or 0.0)
    if adx_value >= 25.0:
        score += 10.0

    distance_vwap = float(row.get("distance_from_vwap_pct") or 0.0)
    if abs(distance_vwap) <= 1.5:
        score += 5.0

    if bool(row.get("opening_range_breakout")):
        score += 10.0
    elif bool(row.get("opening_range_breakdown")):
        score -= 10.0
    return max(0.0, min(100.0, score))


def add_feature_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV plus all point-in-time technical features."""
    return add_indicators(df)


def latest_feature_snapshot(
    df: pd.DataFrame,
    *,
    symbol: str,
    instrument_key: str,
    timeframe: str = "1m",
    support_window: int = 20,
) -> dict:
    """Build one serializable snapshot from the latest available candle only."""
    enriched = add_feature_engine(df)
    if enriched.empty:
        raise ValueError("Cannot build a feature snapshot from an empty dataframe")

    row = enriched.iloc[-1]
    recent = enriched.iloc[max(0, len(enriched) - support_window - 1):-1]
    support = float(recent["low"].min()) if not recent.empty else float(row["low"])
    resistance = float(recent["high"].max()) if not recent.empty else float(row["high"])
    close = float(row["close"])

    distance_to_support = ((close - support) / support * 100.0) if support else 0.0
    distance_to_resistance = ((resistance - close) / resistance * 100.0) if resistance else 0.0

    event_time = pd.Timestamp(row["timestamp"])
    local_time = event_time.tz_convert("Asia/Kolkata") if event_time.tzinfo else event_time.tz_localize("Asia/Kolkata")

    snapshot = {
        "event_time": event_time.to_pydatetime(),
        "trading_date": local_time.date(),
        "symbol": symbol,
        "instrument_key": instrument_key,
        "timeframe": timeframe,
        "close": close,
        "volume": float(row["volume"]),
        "relative_volume": _json_value(row.get("relative_volume")),
        "vwap": _json_value(row.get("vwap")),
        "rsi14": _json_value(row.get("rsi14")),
        "ema9": _json_value(row.get("ema9")),
        "ema20": _json_value(row.get("ema20")),
        "ema50": _json_value(row.get("ema50")),
        "ema200": _json_value(row.get("ema200")),
        "atr14": _json_value(row.get("atr14")),
        "support": support,
        "resistance": resistance,
        "distance_to_support_pct": distance_to_support,
        "distance_to_resistance_pct": distance_to_resistance,
        "candle_pattern": None,
        "trend": row.get("trend"),
        "breakout": bool(row.get("opening_range_breakout", False)),
        "breakdown": bool(row.get("opening_range_breakdown", False)),
        "feature_score": technical_feature_score(row),
    }

    feature_columns = [
        "adx14", "plus_di14", "minus_di14", "macd", "macd_signal", "macd_histogram",
        "bb_middle", "bb_upper", "bb_lower", "bb_width_pct", "atr_pct", "session_volume",
        "session_high", "session_low", "distance_from_vwap_pct", "opening_range_high",
        "opening_range_low", "opening_range_breakout", "opening_range_breakdown",
    ]
    snapshot["feature_json"] = json.dumps(
        {column: _json_value(row.get(column)) for column in feature_columns},
        sort_keys=True,
        separators=(",", ":"),
    )
    return snapshot
