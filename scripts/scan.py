from __future__ import annotations

import argparse

from config.settings import settings
from intraday_engine.market.upstox import UpstoxREST
from intraday_engine.scanner.service import ScannerConfig, scan_top10


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank the top intraday NSE candidates")
    parser.add_argument("--limit", type=int, default=settings.top_n)
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--min-price", type=float, default=50.0)
    parser.add_argument("--min-avg-traded-value", type=float, default=50_000_000.0)
    parser.add_argument("--news-lookback-hours", type=int, default=settings.news_lookback_hours)
    args = parser.parse_args()

    rows = scan_top10(
        UpstoxREST(),
        ScannerConfig(
            limit=args.limit,
            lookback_days=args.lookback_days,
            minimum_price=args.min_price,
            minimum_avg_daily_traded_value=args.min_avg_traded_value,
            news_lookback_hours=args.news_lookback_hours,
        ),
    )
    if not rows:
        print("No candidates found. Ensure instrument master and historical candles are populated.")
        return

    print("rank symbol score change_pct rvol news reason")
    for row in rows:
        print(
            f"{row['rank']:>4} {row['symbol']:<14} {row['candidate_score']:>5.1f} "
            f"{row['change_pct']:>9.2f}% {row['relative_volume']:>5.2f}x "
            f"{row.get('news_score', 0.0):>+6.2f} {row['reason']}"
        )


if __name__ == "__main__":
    main()
