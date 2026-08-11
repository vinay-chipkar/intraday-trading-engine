from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SignalConfig:
    buy_threshold: float = 60.0
    sell_threshold: float = -60.0
    min_adx: float = 20.0
    min_rvol: float = 1.0
    stop_atr_multiple: float = 1.0
    target_atr_multiple: float = 1.5
    max_distance_from_vwap_pct: float = 2.0
    use_structural_stop: bool = True
    structural_stop_buffer_atr: float = 0.10
    max_structural_stop_atr: float = 2.50


@dataclass(frozen=True)
class TradeSignal:
    action: str
    score: float
    confidence: float
    entry: float
    stop_loss: float | None
    target: float | None
    reward_risk: float | None
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    symbol: str | None = None
    event_time: object | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def side(self) -> str:
        return {"BUY": "LONG", "SELL": "SHORT"}.get(self.action, self.action)


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score(row: dict[str, Any], market_score: float) -> tuple[float, list[str], list[str]]:
    score = 0.0
    reasons: list[str] = []
    blockers: list[str] = []

    trend = str(row.get("trend", "SIDEWAYS"))
    structure = str(row.get("structure_trend", row.get("trend", "SIDEWAYS")))
    close = _number(row, "close")
    ema9 = _number(row, "ema9")
    ema20 = _number(row, "ema20")
    ema50 = _number(row, "ema50")
    vwap = _number(row, "vwap")
    rsi = _number(row, "rsi14", 50.0)
    adx = _number(row, "adx14")
    plus_di = _number(row, "plus_di14")
    minus_di = _number(row, "minus_di14")
    macd_hist = _number(row, "macd_histogram")
    rvol = _number(row, "relative_volume")
    distance_vwap = _number(row, "distance_from_vwap_pct")

    if trend == "UPTREND":
        score += 15
        reasons.append("EMA trend is bullish")
    elif trend == "DOWNTREND":
        score -= 15
        reasons.append("EMA trend is bearish")

    if structure == "UPTREND":
        score += 15
        reasons.append("market structure is bullish")
    elif structure == "DOWNTREND":
        score -= 15
        reasons.append("market structure is bearish")

    if close > ema9 > ema20 > ema50 > 0:
        score += 10
        reasons.append("price is above aligned EMAs")
    elif 0 < close < ema9 < ema20 < ema50:
        score -= 10
        reasons.append("price is below aligned EMAs")

    if vwap > 0 and close > vwap:
        score += 8
        reasons.append("price is above VWAP")
    elif vwap > 0 and close < vwap:
        score -= 8
        reasons.append("price is below VWAP")

    if 52 <= rsi <= 70:
        score += 8
        reasons.append("RSI supports bullish momentum")
    elif 30 <= rsi <= 48:
        score -= 8
        reasons.append("RSI supports bearish momentum")

    if macd_hist > 0:
        score += 7
        reasons.append("MACD histogram is positive")
    elif macd_hist < 0:
        score -= 7
        reasons.append("MACD histogram is negative")

    if adx >= 25:
        score += 8 if plus_di >= minus_di else -8
        reasons.append("ADX confirms directional strength")
    elif adx < 15:
        blockers.append("trend strength is too weak")

    if rvol >= 1.5:
        score += 7 if close >= vwap else -7
        reasons.append("relative volume is elevated")
    elif rvol > 0 and rvol < 0.7:
        blockers.append("relative volume is weak")

    if row.get("opening_range_breakout"):
        score += 10
        reasons.append("opening range breakout")
    elif row.get("opening_range_breakdown"):
        score -= 10
        reasons.append("opening range breakdown")

    if row.get("hammer") or row.get("bullish_engulfing") or row.get("morning_star"):
        score += 8
        reasons.append("bullish candlestick confirmation")
    if row.get("shooting_star") or row.get("bearish_engulfing") or row.get("evening_star"):
        score -= 8
        reasons.append("bearish candlestick confirmation")

    if row.get("double_bottom"):
        score += 8
        reasons.append("double bottom structure")
    if row.get("double_top"):
        score -= 8
        reasons.append("double top structure")

    if abs(distance_vwap) > 2.5:
        blockers.append("price is extended from VWAP")

    score += max(-10.0, min(10.0, float(market_score)))
    return max(-100.0, min(100.0, score)), reasons, blockers


def _structural_stop(
    row: dict[str, Any],
    *,
    side: str,
    entry: float,
    atr: float,
    config: SignalConfig,
) -> tuple[float | None, str | None]:
    atr_stop = (
        entry - config.stop_atr_multiple * atr
        if side == "LONG"
        else entry + config.stop_atr_multiple * atr
    )
    if not config.use_structural_stop:
        return atr_stop, None

    level_key = "support" if side == "LONG" else "resistance"
    level = _number(row, level_key, float("nan"))
    if level != level:  # NaN
        return atr_stop, None

    buffer = config.structural_stop_buffer_atr * atr
    structural = level - buffer if side == "LONG" else level + buffer
    risk = abs(entry - structural)
    if risk <= 0:
        return None, "structural stop is invalid"
    # A LONG stop must sit strictly below entry (and a SHORT stop strictly above);
    # otherwise abs(entry - structural) still yields a positive "risk" for a stop
    # that is on the wrong side of entry -- e.g. price has already broken through
    # support/resistance on this same bar, so "level - buffer" (LONG) can land
    # above entry instead of below it.
    if (side == "LONG" and structural > entry) or (side == "SHORT" and structural < entry):
        return None, "structural stop is on the wrong side of entry"
    if risk > config.max_structural_stop_atr * atr:
        return None, "structural stop exceeds maximum risk"
    return structural, "structural stop" 


def generate_signal(
    row: dict[str, Any],
    *,
    market_score: float = 0.0,
    config: SignalConfig | None = None,
    symbol: str | None = None,
    event_time: object | None = None,
) -> TradeSignal:
    """Generate the single deterministic research signal from one point-in-time row."""
    cfg = config or SignalConfig()
    entry = _number(row, "close")
    atr = _number(row, "atr14")
    score, reasons, blockers = _score(row, market_score)

    if entry <= 0:
        blockers.append("invalid entry price")
    if atr <= 0:
        blockers.append("ATR is unavailable")

    action = "NO_TRADE"
    stop_loss = None
    target = None
    reward_risk = None

    if not blockers and score >= cfg.buy_threshold:
        action = "BUY"
    elif not blockers and score <= cfg.sell_threshold:
        action = "SELL"

    if action != "NO_TRADE":
        side = "LONG" if action == "BUY" else "SHORT"
        stop_loss, stop_reason = _structural_stop(
            row, side=side, entry=entry, atr=atr, config=cfg
        )
        if stop_reason:
            reasons.append(stop_reason) if stop_reason == "structural stop" else blockers.append(stop_reason)
        if stop_loss is None:
            action = "NO_TRADE"
        else:
            risk = abs(entry - stop_loss)
            target = entry + cfg.target_atr_multiple * risk if side == "LONG" else entry - cfg.target_atr_multiple * risk
            reward = abs(target - entry)
            reward_risk = reward / risk if risk else None

    confidence = min(100.0, abs(score))
    return TradeSignal(
        action=action,
        score=round(score, 4),
        confidence=round(confidence, 4),
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        reward_risk=reward_risk,
        reasons=tuple(reasons),
        blockers=tuple(blockers),
        symbol=symbol,
        event_time=event_time,
    )
