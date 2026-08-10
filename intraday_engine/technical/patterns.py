from __future__ import annotations

import pandas as pd


PATTERN_COLUMNS = [
    "hammer", "inverted_hammer", "doji", "shooting_star", "marubozu",
    "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star",
    "inside_bar", "outside_bar", "three_white_soldiers", "three_black_crows",
]


def _body(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def _range(row: pd.Series) -> float:
    return max(float(row["high"]) - float(row["low"]), 1e-12)


def _upper_wick(row: pd.Series) -> float:
    return float(row["high"]) - max(float(row["open"]), float(row["close"]))


def _lower_wick(row: pd.Series) -> float:
    return min(float(row["open"]), float(row["close"])) - float(row["low"])


def _bullish(row: pd.Series) -> bool:
    return float(row["close"]) > float(row["open"])


def _bearish(row: pd.Series) -> bool:
    return float(row["close"]) < float(row["open"])


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic candlestick pattern flags using current/prior candles only."""
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)
    for name in PATTERN_COLUMNS:
        out[name] = False

    for i in range(len(out)):
        cur = out.iloc[i]
        rng = _range(cur)
        body = _body(cur)
        upper = _upper_wick(cur)
        lower = _lower_wick(cur)
        body_ratio = body / rng

        out.at[i, "doji"] = body_ratio <= 0.10
        out.at[i, "hammer"] = lower >= max(body * 2.0, rng * 0.45) and upper <= rng * 0.20 and body_ratio <= 0.40
        out.at[i, "inverted_hammer"] = upper >= max(body * 2.0, rng * 0.45) and lower <= rng * 0.20 and body_ratio <= 0.40
        out.at[i, "shooting_star"] = out.at[i, "inverted_hammer"] and _bearish(cur)
        out.at[i, "marubozu"] = body_ratio >= 0.80 and upper <= rng * 0.10 and lower <= rng * 0.10

        if i >= 1:
            prev = out.iloc[i - 1]
            out.at[i, "bullish_engulfing"] = (
                _bearish(prev) and _bullish(cur)
                and float(cur["open"]) <= float(prev["close"])
                and float(cur["close"]) >= float(prev["open"])
                and body >= _body(prev)
            )
            out.at[i, "bearish_engulfing"] = (
                _bullish(prev) and _bearish(cur)
                and float(cur["open"]) >= float(prev["close"])
                and float(cur["close"]) <= float(prev["open"])
                and body >= _body(prev)
            )
            out.at[i, "inside_bar"] = float(cur["high"]) <= float(prev["high"]) and float(cur["low"]) >= float(prev["low"])
            out.at[i, "outside_bar"] = float(cur["high"]) >= float(prev["high"]) and float(cur["low"]) <= float(prev["low"])

        if i >= 2:
            first = out.iloc[i - 2]
            middle = out.iloc[i - 1]
            first_body = _body(first)
            middle_body = _body(middle)
            first_midpoint = (float(first["open"]) + float(first["close"])) / 2.0
            out.at[i, "morning_star"] = (
                _bearish(first)
                and middle_body <= first_body * 0.50
                and _bullish(cur)
                and float(cur["close"]) > first_midpoint
            )
            out.at[i, "evening_star"] = (
                _bullish(first)
                and middle_body <= first_body * 0.50
                and _bearish(cur)
                and float(cur["close"]) < first_midpoint
            )

            a, b, c = out.iloc[i - 2], out.iloc[i - 1], out.iloc[i]
            out.at[i, "three_white_soldiers"] = (
                _bullish(a) and _bullish(b) and _bullish(c)
                and float(b["close"]) > float(a["close"])
                and float(c["close"]) > float(b["close"])
                and _upper_wick(a) <= _range(a) * 0.25
                and _upper_wick(b) <= _range(b) * 0.25
                and _upper_wick(c) <= _range(c) * 0.25
            )
            out.at[i, "three_black_crows"] = (
                _bearish(a) and _bearish(b) and _bearish(c)
                and float(b["close"]) < float(a["close"])
                and float(c["close"]) < float(b["close"])
                and _lower_wick(a) <= _range(a) * 0.25
                and _lower_wick(b) <= _range(b) * 0.25
                and _lower_wick(c) <= _range(c) * 0.25
            )

    return out


def latest_pattern_names(df: pd.DataFrame) -> list[str]:
    """Return active pattern names on the latest candle in deterministic order."""
    enriched = detect_candlestick_patterns(df)
    if enriched.empty:
        return []
    row = enriched.iloc[-1]
    return [column for column in PATTERN_COLUMNS if bool(row[column])]
