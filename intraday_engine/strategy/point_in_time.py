from __future__ import annotations

import pandas as pd

from intraday_engine.signals.engine import SignalConfig, TradeSignal, generate_signal
from intraday_engine.strategy.features import enrich


def _confirmed_pivots(df: pd.DataFrame, left: int, right: int) -> tuple[pd.Series, pd.Series]:
    """Return pivot prices on the first bar where their confirmation exists."""
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")
    high_candidate = df["high"].eq(df["high"].rolling(left + right + 1, center=True).max())
    low_candidate = df["low"].eq(df["low"].rolling(left + right + 1, center=True).min())
    return df["high"].where(high_candidate).shift(right), df["low"].where(low_candidate).shift(right)


def pivot_structure(confirmed_pivot_high: pd.Series, confirmed_pivot_low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Walk pivots confirmed so far into a causal trend/support/resistance series.

    Shared by `enrich_point_in_time` (the live signal path) and any research/backtest
    code that wants the same point-in-time market-structure definition instead of
    reimplementing it from a different, non-causal proxy (e.g. an EMA-order trend).
    """
    last_high = last_low = previous_high = previous_low = None
    trends: list[str] = []
    supports: list[float | None] = []
    resistances: list[float | None] = []

    for high, low, price in zip(confirmed_pivot_high, confirmed_pivot_low, close):
        if pd.notna(high):
            previous_high, last_high = last_high, float(high)
        if pd.notna(low):
            previous_low, last_low = last_low, float(low)

        support = last_low if last_low is not None and last_low <= float(price) else None
        resistance = last_high if last_high is not None and last_high >= float(price) else None

        if previous_high is not None and previous_low is not None and last_high is not None and last_low is not None:
            if last_high > previous_high and last_low > previous_low:
                trend = "UPTREND"
            elif last_high < previous_high and last_low < previous_low:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
        else:
            trend = "SIDEWAYS"

        supports.append(support)
        resistances.append(resistance)
        trends.append(trend)

    return pd.DataFrame(
        {"trend": trends, "support": supports, "resistance": resistances}, index=confirmed_pivot_high.index
    )


def enrich_point_in_time(df: pd.DataFrame, *, pivot_left: int = 3, pivot_right: int = 3, breakout_lookback: int = 20) -> pd.DataFrame:
    """Build features without using information after each row's timestamp."""
    if df.empty:
        return df.copy()

    out = enrich(df.copy().reset_index(drop=True))
    confirmed_high, confirmed_low = _confirmed_pivots(out, pivot_left, pivot_right)
    out["confirmed_pivot_high"] = confirmed_high
    out["confirmed_pivot_low"] = confirmed_low

    structure = pivot_structure(out["confirmed_pivot_high"], out["confirmed_pivot_low"], out["close"])
    out["support"] = structure["support"]
    out["resistance"] = structure["resistance"]
    out["trend"] = structure["trend"]
    out["distance_to_support_pct"] = (out["close"] - out["support"]) / out["close"] * 100
    out["distance_to_resistance_pct"] = (out["resistance"] - out["close"]) / out["close"] * 100
    return out


def generate_signals(df: pd.DataFrame, *, symbol: str, market_score: float = 0.0, min_score: float = 60.0, pivot_left: int = 3, pivot_right: int = 3) -> list[TradeSignal]:
    """Generate signals exclusively through the canonical signal engine."""
    features = enrich_point_in_time(df, pivot_left=pivot_left, pivot_right=pivot_right)
    config = SignalConfig(buy_threshold=min_score, sell_threshold=-min_score)
    signals: list[TradeSignal] = []

    for _, row in features.iterrows():
        signal = generate_signal(row.to_dict(), market_score=market_score, config=config, symbol=symbol, event_time=row["timestamp"])
        if signal.action in {"BUY", "SELL"}:
            signals.append(signal)
    return signals
