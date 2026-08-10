from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")


def _confirmed_pivots(
    df: pd.DataFrame,
    left: int,
    right: int,
) -> tuple[pd.Series, pd.Series]:
    """Return pivot values only when the pivot has become confirmable.

    A pivot at candle j is written to candle j + right. Therefore the output
    never requires data after the current row and is safe for point-in-time
    features/backtests.
    """
    highs = pd.Series(np.nan, index=df.index, dtype=float)
    lows = pd.Series(np.nan, index=df.index, dtype=float)

    for confirmation_idx in range(left + right, len(df)):
        pivot_idx = confirmation_idx - right
        start = pivot_idx - left
        end = pivot_idx + right + 1
        window_high = df.iloc[start:end]["high"]
        window_low = df.iloc[start:end]["low"]
        pivot_high = float(df.iloc[pivot_idx]["high"])
        pivot_low = float(df.iloc[pivot_idx]["low"])

        if pivot_high >= float(window_high.max()) and (window_high == pivot_high).sum() == 1:
            highs.iloc[confirmation_idx] = pivot_high
        if pivot_low <= float(window_low.min()) and (window_low == pivot_low).sum() == 1:
            lows.iloc[confirmation_idx] = pivot_low

    return highs, lows


def _latest_level(values: list[float], close: float, *, support: bool) -> float:
    if support:
        candidates = [value for value in values if value <= close]
        return max(candidates) if candidates else (max(values) if values else close)

    # Resistance represents the strongest confirmed overhead structural level,
    # so select the highest confirmed pivot above the current close. If price
    # has already broken above all confirmed highs, retain the highest known
    # structural high rather than dropping to an older/lower pivot.
    candidates = [value for value in values if value >= close]
    return max(candidates) if candidates else (max(values) if values else close)


def _classify_structure(pivots: list[tuple[str, float]]) -> str:
    highs = [value for kind, value in pivots if kind == "H"]
    lows = [value for kind, value in pivots if kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return "UNKNOWN"

    higher_high = highs[-1] > highs[-2]
    higher_low = lows[-1] > lows[-2]
    lower_high = highs[-1] < highs[-2]
    lower_low = lows[-1] < lows[-2]

    if higher_high and higher_low:
        return "BULLISH"
    if lower_high and lower_low:
        return "BEARISH"
    return "MIXED"


def add_market_structure(
    df: pd.DataFrame,
    *,
    pivot_left: int = 2,
    pivot_right: int = 2,
    level_lookback: int = 100,
    double_pattern_tolerance: float = 0.003,
    double_pattern_min_bars: int = 3,
) -> pd.DataFrame:
    """Add point-in-time support/resistance and market-structure features.

    Features are based only on pivots that have already been confirmed. The
    current candle itself is never used as an unconfirmed pivot.
    """
    _validate_ohlc(df)
    if pivot_left < 1 or pivot_right < 1:
        raise ValueError("pivot_left and pivot_right must be >= 1")
    if level_lookback < 1:
        raise ValueError("level_lookback must be >= 1")

    out = df.copy().reset_index(drop=True)
    pivot_high, pivot_low = _confirmed_pivots(out, pivot_left, pivot_right)
    out["pivot_high"] = pivot_high
    out["pivot_low"] = pivot_low

    out["support"] = np.nan
    out["resistance"] = np.nan
    out["market_structure"] = "UNKNOWN"
    out["structure_trend"] = "SIDEWAYS"
    out["double_top"] = False
    out["double_bottom"] = False
    out["support_slope"] = np.nan
    out["resistance_slope"] = np.nan

    confirmed: list[tuple[int, str, float]] = []
    for i in range(len(out)):
        if pd.notna(pivot_high.iloc[i]):
            confirmed.append((i, "H", float(pivot_high.iloc[i])))
        if pd.notna(pivot_low.iloc[i]):
            confirmed.append((i, "L", float(pivot_low.iloc[i])))

        recent = [item for item in confirmed if item[0] >= max(0, i - level_lookback + 1)]
        close = float(out.iloc[i]["close"])
        highs = [value for _, kind, value in recent if kind == "H"]
        lows = [value for _, kind, value in recent if kind == "L"]
        out.at[i, "support"] = _latest_level(lows, close, support=True)
        out.at[i, "resistance"] = _latest_level(highs, close, support=False)

        alternating = sorted(recent, key=lambda item: item[0])
        structure_pairs = [(kind, value) for _, kind, value in alternating]
        structure = _classify_structure(structure_pairs[-8:])
        out.at[i, "market_structure"] = structure
        out.at[i, "structure_trend"] = {
            "BULLISH": "UPTREND",
            "BEARISH": "DOWNTREND",
            "MIXED": "SIDEWAYS",
        }.get(structure, "SIDEWAYS")

        recent_highs = [(idx, value) for idx, kind, value in recent if kind == "H"]
        recent_lows = [(idx, value) for idx, kind, value in recent if kind == "L"]
        if len(recent_highs) >= 2:
            (idx1, high1), (idx2, high2) = recent_highs[-2:]
            out.at[i, "double_top"] = (
                idx2 - idx1 >= double_pattern_min_bars
                and abs(high2 - high1) / max(abs(high1), 1e-12) <= double_pattern_tolerance
            )
        if len(recent_lows) >= 2:
            (idx1, low1), (idx2, low2) = recent_lows[-2:]
            out.at[i, "double_bottom"] = (
                idx2 - idx1 >= double_pattern_min_bars
                and abs(low2 - low1) / max(abs(low1), 1e-12) <= double_pattern_tolerance
            )

        if len(recent_lows) >= 2:
            x1, y1 = recent_lows[-2]
            x2, y2 = recent_lows[-1]
            if x2 != x1:
                out.at[i, "support_slope"] = (y2 - y1) / (x2 - x1)
        if len(recent_highs) >= 2:
            x1, y1 = recent_highs[-2]
            x2, y2 = recent_highs[-1]
            if x2 != x1:
                out.at[i, "resistance_slope"] = (y2 - y1) / (x2 - x1)

    return out


def latest_market_structure(df: pd.DataFrame, **kwargs) -> dict:
    """Return the latest confirmed support/resistance structure snapshot."""
    enriched = add_market_structure(df, **kwargs)
    if enriched.empty:
        return {}
    row = enriched.iloc[-1]
    return {
        "support": float(row["support"]),
        "resistance": float(row["resistance"]),
        "pivot_high": None if pd.isna(row["pivot_high"]) else float(row["pivot_high"]),
        "pivot_low": None if pd.isna(row["pivot_low"]) else float(row["pivot_low"]),
        "market_structure": str(row["market_structure"]),
        "structure_trend": str(row["structure_trend"]),
        "double_top": bool(row["double_top"]),
        "double_bottom": bool(row["double_bottom"]),
        "support_slope": None if pd.isna(row["support_slope"]) else float(row["support_slope"]),
        "resistance_slope": None if pd.isna(row["resistance_slope"]) else float(row["resistance_slope"]),
    }
