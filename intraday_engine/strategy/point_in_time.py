from __future__ import annotations

from typing import Iterable

import pandas as pd

from intraday_engine.strategy.features import enrich
from intraday_engine.strategy.scoring import score_long, score_short
from intraday_engine.strategy.signals import Signal, build_signal


def _confirmed_pivots(df: pd.DataFrame, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    """Return pivot values only when the confirming right-side bars exist."""
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")

    high_candidate = df["high"].eq(
        df["high"].rolling(left + right + 1, center=True).max()
    )
    low_candidate = df["low"].eq(
        df["low"].rolling(left + right + 1, center=True).min()
    )

    # A pivot at t-right becomes known at t. The shift moves the candidate
    # forward by right bars, so no feature at t can depend on bars after t.
    confirmed_high = df["high"].where(high_candidate).shift(right)
    confirmed_low = df["low"].where(low_candidate).shift(right)
    return confirmed_high, confirmed_low


def enrich_point_in_time(
    df: pd.DataFrame,
    *,
    pivot_left: int = 3,
    pivot_right: int = 3,
    breakout_lookback: int = 20,
) -> pd.DataFrame:
    """Build features without using information after each row's timestamp.

    This is the historical/backtest feature path. It intentionally does not
    use the existing full-frame support/resistance helper because that helper
    is designed for the latest snapshot and uses centered pivots directly.
    """
    if df.empty:
        return df.copy()

    out = df.copy().reset_index(drop=True)
    out = enrich(out)

    # Replace the latest-snapshot structure fields with causal, confirmed
    # pivots. Breakout detection in enrich() already uses shift(1).
    confirmed_high, confirmed_low = _confirmed_pivots(
        out, pivot_left, pivot_right
    )
    out["confirmed_pivot_high"] = confirmed_high
    out["confirmed_pivot_low"] = confirmed_low

    last_high = None
    last_low = None
    previous_high = None
    previous_low = None
    trends: list[str] = []
    supports: list[float | None] = []
    resistances: list[float | None] = []

    for high, low, close in zip(
        out["confirmed_pivot_high"],
        out["confirmed_pivot_low"],
        out["close"],
    ):
        if pd.notna(high):
            previous_high = last_high
            last_high = float(high)
        if pd.notna(low):
            previous_low = last_low
            last_low = float(low)

        if last_low is not None and last_low <= float(close):
            support = last_low
        else:
            support = None

        if last_high is not None and last_high >= float(close):
            resistance = last_high
        else:
            resistance = None

        if (
            previous_high is not None
            and previous_low is not None
            and last_high is not None
            and last_low is not None
            and last_high > previous_high
            and last_low > previous_low
        ):
            trend = "UPTREND"
        elif (
            previous_high is not None
            and previous_low is not None
            and last_high is not None
            and last_low is not None
            and last_high < previous_high
            and last_low < previous_low
        ):
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"

        supports.append(support)
        resistances.append(resistance)
        trends.append(trend)

    out["support"] = supports
    out["resistance"] = resistances
    out["trend"] = trends
    out["distance_to_support_pct"] = (
        (out["close"] - out["support"]) / out["close"] * 100
    )
    out["distance_to_resistance_pct"] = (
        (out["resistance"] - out["close"]) / out["close"] * 100
    )
    return out


def generate_signals(
    df: pd.DataFrame,
    *,
    symbol: str,
    market_score: float = 0.0,
    min_score: float = 60.0,
    pivot_left: int = 3,
    pivot_right: int = 3,
) -> list[Signal]:
    """Generate deterministic rule-based signals from point-in-time rows."""
    features = enrich_point_in_time(
        df,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
    )
    signals: list[Signal] = []

    for _, row in features.iterrows():
        values = row.to_dict()
        long_score, long_reasons = score_long(values, market_score)
        short_score, short_reasons = score_short(values, market_score)

        if long_score >= min_score and long_score > short_score:
            signal = build_signal(symbol, row, long_score, long_reasons, "LONG")
        elif short_score >= min_score and short_score > long_score:
            signal = build_signal(symbol, row, short_score, short_reasons, "SHORT")
        else:
            signal = None

        if signal is not None:
            signals.append(signal)

    return signals
