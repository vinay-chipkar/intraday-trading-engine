from __future__ import annotations

import json

import pandas as pd


PATTERN_COLUMNS = [
    "hammer", "hanging_man", "inverted_hammer", "shooting_star", "doji", "marubozu",
    "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star",
    "inside_bar", "outside_bar", "three_white_soldiers", "three_black_crows",
]


def _body(r: pd.Series) -> float:
    return abs(float(r.open) - float(r.close))


def _range(r: pd.Series) -> float:
    return max(float(r.high) - float(r.low), 1e-12)


def _upper(r: pd.Series) -> float:
    return float(r.high) - max(float(r.open), float(r.close))


def _lower(r: pd.Series) -> float:
    return min(float(r.open), float(r.close)) - float(r.low)


def _bullish(r: pd.Series) -> bool:
    return float(r.close) > float(r.open)


def _bearish(r: pd.Series) -> bool:
    return float(r.close) < float(r.open)


def classify_candle(r: pd.Series) -> str:
    """Classify one candle without using any future information."""
    body = _body(r)
    rng = _range(r)
    upper = _upper(r)
    lower = _lower(r)

    if body / rng <= 0.10:
        return "DOJI"
    if lower >= max(2.0 * body, 0.45 * rng) and upper <= 0.20 * rng and body / rng <= 0.40:
        return "HAMMER" if _bullish(r) else "HANGING_MAN"
    if upper >= max(2.0 * body, 0.45 * rng) and lower <= 0.20 * rng and body / rng <= 0.40:
        return "INVERTED_HAMMER" if _bullish(r) else "SHOOTING_STAR"
    if body / rng >= 0.80 and upper <= 0.10 * rng and lower <= 0.10 * rng:
        return "BULLISH_MARUBOZU" if _bullish(r) else "BEARISH_MARUBOZU"
    return "BULLISH" if _bullish(r) else "BEARISH"


def _pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
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
        upper = _upper(cur)
        lower = _lower(cur)
        ratio = body / rng

        out.at[i, "doji"] = ratio <= 0.10
        out.at[i, "hammer"] = lower >= max(body * 2.0, rng * 0.45) and upper <= rng * 0.20 and ratio <= 0.40
        out.at[i, "hanging_man"] = out.at[i, "hammer"] and _bearish(cur)
        out.at[i, "inverted_hammer"] = upper >= max(body * 2.0, rng * 0.45) and lower <= rng * 0.20 and ratio <= 0.40
        out.at[i, "shooting_star"] = out.at[i, "inverted_hammer"] and _bearish(cur)
        out.at[i, "marubozu"] = ratio >= 0.80 and upper <= rng * 0.10 and lower <= rng * 0.10

        if i >= 1:
            prev = out.iloc[i - 1]
            out.at[i, "bullish_engulfing"] = (
                _bearish(prev) and _bullish(cur)
                and float(cur.open) <= float(prev.close)
                and float(cur.close) >= float(prev.open)
                and body >= _body(prev)
            )
            out.at[i, "bearish_engulfing"] = (
                _bullish(prev) and _bearish(cur)
                and float(cur.open) >= float(prev.close)
                and float(cur.close) <= float(prev.open)
                and body >= _body(prev)
            )
            out.at[i, "inside_bar"] = float(cur.high) <= float(prev.high) and float(cur.low) >= float(prev.low)
            out.at[i, "outside_bar"] = float(cur.high) >= float(prev.high) and float(cur.low) <= float(prev.low)

        if i >= 2:
            first, middle = out.iloc[i - 2], out.iloc[i - 1]
            first_body = _body(first)
            middle_body = _body(middle)
            midpoint = (float(first.open) + float(first.close)) / 2.0
            out.at[i, "morning_star"] = (
                _bearish(first) and middle_body <= first_body * 0.50
                and _bullish(cur) and float(cur.close) > midpoint
            )
            out.at[i, "evening_star"] = (
                _bullish(first) and middle_body <= first_body * 0.50
                and _bearish(cur) and float(cur.close) < midpoint
            )

            a, b, c = out.iloc[i - 2], out.iloc[i - 1], out.iloc[i]
            out.at[i, "three_white_soldiers"] = (
                _bullish(a) and _bullish(b) and _bullish(c)
                and float(b.close) > float(a.close) and float(c.close) > float(b.close)
                and _upper(a) <= _range(a) * 0.25
                and _upper(b) <= _range(b) * 0.25
                and _upper(c) <= _range(c) * 0.25
            )
            out.at[i, "three_black_crows"] = (
                _bearish(a) and _bearish(b) and _bearish(c)
                and float(b.close) < float(a.close) and float(c.close) < float(b.close)
                and _lower(a) <= _range(a) * 0.25
                and _lower(b) <= _range(b) * 0.25
                and _lower(c) <= _range(c) * 0.25
            )

    return out


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a primary candle label plus transparent pattern flags."""
    out = _pattern_flags(df)
    out["candle_pattern"] = [classify_candle(row) for _, row in out.iterrows()]

    priority = [
        "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star",
        "three_white_soldiers", "three_black_crows", "hanging_man", "hammer",
        "shooting_star", "inverted_hammer", "doji", "marubozu", "inside_bar", "outside_bar",
    ]
    labels = {
        "bullish_engulfing": "BULLISH_ENGULFING", "bearish_engulfing": "BEARISH_ENGULFING",
        "morning_star": "MORNING_STAR", "evening_star": "EVENING_STAR",
        "three_white_soldiers": "THREE_WHITE_SOLDIERS", "three_black_crows": "THREE_BLACK_CROWS",
        "hammer": "HAMMER", "hanging_man": "HANGING_MAN", "inverted_hammer": "INVERTED_HAMMER",
        "shooting_star": "SHOOTING_STAR", "doji": "DOJI", "marubozu": "MARUBOZU",
        "inside_bar": "INSIDE_BAR", "outside_bar": "OUTSIDE_BAR",
    }
    for i in range(len(out)):
        active = [name for name in priority if bool(out.at[i, name])]
        if active:
            out.at[i, "candle_pattern"] = labels[active[0]]
    out["pattern_flags"] = out.apply(
        lambda row: json.dumps({name: bool(row[name]) for name in PATTERN_COLUMNS}, sort_keys=True, separators=(",", ":")),
        axis=1,
    )
    return out


def latest_pattern_names(df: pd.DataFrame) -> list[str]:
    """Return all detected pattern names on the latest candle."""
    enriched = add_candle_patterns(df)
    if enriched.empty:
        return []
    row = enriched.iloc[-1]
    return [name for name in PATTERN_COLUMNS if bool(row[name])]
