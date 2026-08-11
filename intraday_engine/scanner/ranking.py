from __future__ import annotations

from typing import Iterable

SESSION_MINUTES = 375  # NSE cash session: 09:15-15:30 IST.


def _percentile(values: list[float], value: float) -> float:
    clean = sorted(v for v in values if v == v)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return 1.0
    lower = sum(v < value for v in clean)
    equal = sum(v == value for v in clean)
    return (lower + 0.5 * equal) / len(clean)


def _rvol_score(rvol: float) -> float:
    if rvol >= 3.0:
        return 30.0
    if rvol >= 2.0:
        return 25.0
    if rvol >= 1.5:
        return 20.0
    if rvol >= 1.0:
        return 12.0
    if rvol >= 0.75:
        return 6.0
    return 0.0


def _momentum_score(change_pct: float, regime_score: float) -> float:
    if regime_score >= 5:
        directional = max(change_pct, 0.0)
    elif regime_score <= -5:
        directional = max(-change_pct, 0.0)
    else:
        directional = abs(change_pct)
    return min(20.0, directional * 5.0)


def _volatility_score(day_range_pct: float) -> float:
    if 1.0 <= day_range_pct <= 4.0:
        return 15.0
    if 0.5 <= day_range_pct < 1.0 or 4.0 < day_range_pct <= 6.0:
        return 9.0
    if 0.25 <= day_range_pct < 0.5 or 6.0 < day_range_pct <= 8.0:
        return 4.0
    return 0.0


def _news_component(news_score: float, market_score: float) -> tuple[float, str | None]:
    news_score = max(-1.0, min(1.0, news_score))
    if abs(news_score) < 0.05:
        return 0.0, None
    if market_score >= 5:
        component = news_score * 10.0
    elif market_score <= -5:
        component = -news_score * 10.0
    else:
        component = abs(news_score) * 5.0
    if component >= 2.0:
        return min(10.0, component), "NEWS_SUPPORT"
    if component <= -2.0:
        return max(-10.0, component), "NEWS_CONFLICT"
    return component, "NEWS_WEAK"


def rank_candidates(rows: Iterable[dict], market_score: float = 0.0, limit: int | None = 10) -> list[dict]:
    """Score and optionally rank/truncate intraday candidates.

    Incomplete historical data is retained for research, but cannot receive
    normal RVOL/liquidity scoring or outrank data-complete candidates.
    Pass ``limit=None`` to return the complete universe.
    """
    rows = [dict(row) for row in rows]
    if not rows:
        return []

    max_elapsed = max(
        1.0,
        min(max(float(row.get("elapsed_session_minutes", 1.0)) for row in rows), SESSION_MINUTES),
    )
    fraction = max_elapsed / SESSION_MINUTES
    liquidity_values = [
        float(row.get("avg_daily_traded_value") or 0.0)
        for row in rows
        if float(row.get("avg_daily_traded_value") or 0.0) > 0
    ]

    ranked: list[dict] = []
    for row in rows:
        ltp = float(row.get("ltp") or 0.0)
        previous_close = float(row.get("previous_close") or 0.0)
        volume = float(row.get("cumulative_volume") or 0.0)
        avg_volume = float(row.get("avg_daily_volume") or 0.0)
        avg_value = float(row.get("avg_daily_traded_value") or 0.0)
        day_high = float(row.get("day_high") or ltp)
        day_low = float(row.get("day_low") or ltp)
        news_score = float(row.get("news_score") or 0.0)
        history_days = int(row.get("history_days") or 0)
        required_history_days = int(row.get("required_history_days") or 20)
        history_complete = history_days >= required_history_days

        change_pct = row.get("change_pct")
        if change_pct is None and previous_close:
            change_pct = (ltp / previous_close - 1.0) * 100.0
        change_pct = float(change_pct or 0.0)

        expected_volume = avg_volume * fraction if history_complete else 0.0
        rvol = volume / expected_volume if expected_volume > 0 else 0.0
        day_range_pct = ((day_high - day_low) / ltp * 100.0) if ltp > 0 else 0.0
        liquidity_pct = _percentile(liquidity_values, avg_value) if history_complete else 0.0
        liquidity_score = 20.0 * liquidity_pct
        news_component, news_reason = _news_component(news_score, market_score)
        score = liquidity_score + _rvol_score(rvol) + _momentum_score(change_pct, market_score) + _volatility_score(day_range_pct) + news_component

        reasons: list[str] = []
        if liquidity_pct >= 0.75:
            reasons.append("HIGH_LIQUIDITY")
        if rvol >= 1.5:
            reasons.append(f"RVOL_{rvol:.2f}X")
        if market_score >= 5 and change_pct > 0:
            reasons.append("BULL_REGIME_ALIGNMENT")
        elif market_score <= -5 and change_pct < 0:
            reasons.append("BEAR_REGIME_ALIGNMENT")
        elif abs(market_score) < 5 and abs(change_pct) >= 0.5:
            reasons.append("MOMENTUM")
        if 1.0 <= day_range_pct <= 4.0:
            reasons.append("HEALTHY_INTRADAY_RANGE")
        if news_reason:
            reasons.append(news_reason)
        if not history_complete:
            reasons.append("DATA_INCOMPLETE")
            # Keep incomplete rows visible in the research universe, but cap
            # their score so they cannot displace data-complete candidates.
            score = min(score, 9.999)

        ranked.append({
            **row,
            "change_pct": round(change_pct, 6),
            "relative_volume": round(rvol, 6),
            "day_range_pct": round(day_range_pct, 6),
            "liquidity_percentile": round(liquidity_pct, 6),
            "news_score": round(news_score, 6),
            "history_complete": history_complete,
            "data_quality": "COMPLETE" if history_complete else ("PARTIAL" if history_days > 0 else "MISSING_HISTORY"),
            "rvol_valid": history_complete,
            "liquidity_valid": history_complete,
            "candidate_score": round(max(0.0, min(100.0, score)), 6),
            "reason": ",".join(reasons) if reasons else "NO_STRONG_FACTOR",
        })

    ranked.sort(key=lambda r: (-r["candidate_score"], -abs(r["change_pct"]), r["symbol"]))
    output = ranked if limit is None else ranked[:limit]
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output
