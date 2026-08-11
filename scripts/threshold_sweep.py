from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from config.settings import settings
from intraday_engine.research.threshold_sensitivity import run_threshold_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run point-in-time signal threshold sensitivity research")
    parser.add_argument("--days", type=int, default=30, help="Most recent calendar-day window to research")
    parser.add_argument("--thresholds", default="40,45,50,55,60,65,70")
    parser.add_argument("--max-holding", type=int, default=30)
    parser.add_argument("--slippage", type=float, default=settings.paper_slippage_points)
    parser.add_argument("--output", default="data/threshold_sweep.csv")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    con = duckdb.connect(settings.duckdb_path, read_only=True)
    frame = con.sql("""
        SELECT timestamp, symbol, open, high, low, close, volume
        FROM candles
        WHERE interval = '1m'
          AND timestamp >= now() - (? * INTERVAL '1 day')
        ORDER BY symbol, timestamp
    """, params=[args.days]).df()
    con.close()
    if frame.empty:
        raise SystemExit("No 1m candles found for the requested window")

    print(f"Threshold sweep window: {frame.timestamp.min()} -> {frame.timestamp.max()}")
    print(f"Symbols: {frame.symbol.nunique()}  rows: {len(frame):,}")
    result = run_threshold_sweep(
        frame,
        thresholds,
        max_holding_bars=args.max_holding,
        slippage_points=args.slippage,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    display_cols = [
        "threshold", "train_trades", "train_win_rate_pct", "train_profit_factor",
        "train_expectancy_r", "oos_trades", "oos_win_rate_pct", "oos_profit_factor",
        "oos_expectancy_r", "oos_max_drawdown_points",
    ]
    print("===== THRESHOLD SENSITIVITY =====")
    print(result[display_cols].to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
