from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from intraday_engine.backtest.engine import BacktestResult, backtest_signals
from intraday_engine.signals.engine import TradeSignal
from intraday_engine.strategy.point_in_time import enrich_point_in_time, generate_signals


FeaturePredicate = Callable[[pd.Series, str], bool]


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: FeaturePredicate


def _finite_bool(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _aligned_emas(row: pd.Series, side: str) -> bool:
    close = float(row.get("close", float("nan")))
    ema9 = float(row.get("ema9", float("nan")))
    ema20 = float(row.get("ema20", float("nan")))
    ema50 = float(row.get("ema50", float("nan")))
    if not all(pd.notna(x) for x in (close, ema9, ema20, ema50)):
        return False
    if side == "LONG":
        return close > ema9 > ema20 > ema50
    return close < ema9 < ema20 < ema50


def _trend(row: pd.Series, side: str) -> bool:
    return str(row.get("trend", "SIDEWAYS")) == ("UPTREND" if side == "LONG" else "DOWNTREND")


def _vwap(row: pd.Series, side: str) -> bool:
    close = row.get("close")
    vwap = row.get("vwap")
    if pd.isna(close) or pd.isna(vwap):
        return False
    return float(close) > float(vwap) if side == "LONG" else float(close) < float(vwap)


def _adx(row: pd.Series, side: str) -> bool:
    adx = row.get("adx14")
    plus_di = row.get("plus_di14")
    minus_di = row.get("minus_di14")
    if any(pd.isna(x) for x in (adx, plus_di, minus_di)):
        return False
    if float(adx) < 25:
        return False
    return float(plus_di) >= float(minus_di) if side == "LONG" else float(minus_di) > float(plus_di)


def _macd(row: pd.Series, side: str) -> bool:
    value = row.get("macd_histogram")
    if pd.isna(value):
        return False
    return float(value) > 0 if side == "LONG" else float(value) < 0


def _not_extended(row: pd.Series, side: str) -> bool:
    value = row.get("distance_from_vwap_pct")
    if pd.isna(value):
        return False
    return abs(float(value)) <= 2.0


def default_filter_specs() -> tuple[FilterSpec, ...]:
    """Small, pre-declared research set; deliberately avoids brute-force optimization."""
    return (
        FilterSpec("baseline", lambda row, side: True),
        FilterSpec("trend_vwap", lambda row, side: _trend(row, side) and _vwap(row, side)),
        FilterSpec(
            "trend_vwap_ema",
            lambda row, side: _trend(row, side) and _vwap(row, side) and _aligned_emas(row, side),
        ),
        FilterSpec(
            "trend_vwap_adx",
            lambda row, side: _trend(row, side) and _vwap(row, side) and _adx(row, side),
        ),
        FilterSpec(
            "trend_vwap_macd",
            lambda row, side: _trend(row, side) and _vwap(row, side) and _macd(row, side),
        ),
        FilterSpec(
            "trend_vwap_ema_adx",
            lambda row, side: _trend(row, side)
            and _vwap(row, side)
            and _aligned_emas(row, side)
            and _adx(row, side),
        ),
        FilterSpec(
            "trend_vwap_ema_macd",
            lambda row, side: _trend(row, side)
            and _vwap(row, side)
            and _aligned_emas(row, side)
            and _macd(row, side),
        ),
        FilterSpec(
            "trend_vwap_ema_adx_macd",
            lambda row, side: _trend(row, side)
            and _vwap(row, side)
            and _aligned_emas(row, side)
            and _adx(row, side)
            and _macd(row, side),
        ),
        FilterSpec(
            "trend_vwap_ema_adx_macd_not_extended",
            lambda row, side: _trend(row, side)
            and _vwap(row, side)
            and _aligned_emas(row, side)
            and _adx(row, side)
            and _macd(row, side)
            and _not_extended(row, side),
        ),
    )


def build_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Return the point-in-time feature frame used for research."""
    return enrich_point_in_time(bars.sort_values("timestamp").reset_index(drop=True))


def prepare_research(
    bars: pd.DataFrame,
    *,
    symbol: str,
    min_score: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[TradeSignal]]:
    """Prepare ordered bars, causal features and signals once per symbol.

    Research evaluates several filters over the same feature/signal stream. Keeping
    this preparation outside the per-filter loop avoids recomputing the expensive
    point-in-time feature frame and signal list for every filter.
    """
    ordered = bars.sort_values("timestamp").reset_index(drop=True).copy()
    ordered["symbol"] = symbol
    features = build_feature_frame(ordered)
    signals = generate_signals(ordered, symbol=symbol, min_score=min_score)
    return ordered, features, signals


def filter_signals(
    signals: list[TradeSignal],
    features: pd.DataFrame,
    predicate: FeaturePredicate,
    *,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> list[TradeSignal]:
    """Filter completed-bar signals using only features at signal time."""
    by_time = {
        pd.Timestamp(row["timestamp"]): row
        for _, row in features.iterrows()
    }
    selected: list[TradeSignal] = []
    for signal in signals:
        if signal.event_time is None:
            continue
        event_time = pd.Timestamp(signal.event_time)
        if start_time is not None and event_time < start_time:
            continue
        if end_time is not None and event_time >= end_time:
            continue
        row = by_time.get(event_time)
        if row is not None and predicate(row, signal.side):
            selected.append(signal)
    return selected


def evaluate_filter(
    bars: pd.DataFrame,
    *,
    symbol: str,
    spec: FilterSpec,
    min_score: float = 60.0,
    max_holding_bars: int = 30,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    features: pd.DataFrame | None = None,
    signals: list[TradeSignal] | None = None,
) -> BacktestResult:
    """Backtest one declared filter without changing the production signal engine.

    ``features`` and ``signals`` may be supplied by callers evaluating multiple
    filters over the same bars. When omitted, this retains the original standalone
    behavior and prepares them internally.
    """
    ordered = bars.sort_values("timestamp").reset_index(drop=True).copy()
    ordered["symbol"] = symbol
    if features is None:
        features = build_feature_frame(ordered)
    if signals is None:
        signals = generate_signals(ordered, symbol=symbol, min_score=min_score)
    selected = filter_signals(
        signals,
        features,
        spec.predicate,
        start_time=start_time,
        end_time=end_time,
    )
    return backtest_signals(selected, ordered, max_holding_bars=max_holding_bars)
