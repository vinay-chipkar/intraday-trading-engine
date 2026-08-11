from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from config.settings import settings
from intraday_engine.backtest.engine import backtest_signals
from intraday_engine.research.threshold_sensitivity import _metrics, _signals_by_threshold
from intraday_engine.technical.indicators import add_indicators
from intraday_engine.patterns.candles import add_candle_patterns


def _prepare_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("timestamp").reset_index(drop=True).copy()
    out = add_indicators(out)
    out = add_candle_patterns(out)
    out["support"] = out["low"].shift(1).rolling(20, min_periods=20).min()
    out["resistance"] = out["high"].shift(1).rolling(20, min_periods=20).max()
    out["structure_trend"] = out["trend"]
    out["double_bottom"] = False
    out["double_top"] = False
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast point-in-time threshold sensitivity research")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--thresholds", default="40,45,50,55,60,65,70")
    parser.add_argument("--max-holding", type=int, default=30)
    parser.add_argument("--slippage", type=float, default=settings.paper_slippage_points)
    parser.add_argument("--output", default="data/threshold_sweep.csv")
    args = parser.parse_args()

    thresholds = tuple(float(x.strip()) for x in args.thresholds.split(",") if x.strip())
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

    print(f"Threshold sweep window: {frame.timestamp.min()} -> {frame.timestamp.max()}", flush=True)
    print(f"Symbols: {frame.symbol.nunique()}  rows: {len(frame):,}", flush=True)

    prepared = []
    for symbol, group in frame.groupby("symbol", sort=True):
        print(f"Preparing {symbol} ({len(group):,} bars)", flush=True)
        prepared.append(_prepare_symbol(group))
    prepared_frame = pd.concat(prepared, ignore_index=True)

    dates = sorted(pd.to_datetime(prepared_frame["timestamp"]).dt.date.unique())
    if len(dates) < 4:
        raise SystemExit("Need at least 4 trading dates for train/OOS research")
    split_date = pd.Timestamp(dates[max(2, int(len(dates) * 0.60))])
    print(f"Train/OOS split: {split_date.date()}", flush=True)

    all_trades = {threshold: [] for threshold in thresholds}
    for symbol, group in prepared_frame.groupby("symbol", sort=True):
        print(f"Backtesting {symbol}", flush=True)
        by_threshold = _signals_by_threshold(group, thresholds)
        for threshold in thresholds:
            result = backtest_signals(
                by_threshold[threshold],
                group,
                max_holding_bars=args.max_holding,
                slippage_points=args.slippage,
            )
            all_trades[threshold].extend(result.trades)

    rows = []
    for threshold in thresholds:
        trades = all_trades[threshold]
        train = [t for t in trades if pd.Timestamp(t.signal_time) < split_date]
        oos = [t for t in trades if pd.Timestamp(t.signal_time) >= split_date]
        rows.append({
            "threshold": threshold,
            "split_date": split_date.date().isoformat(),
            **{f"train_{k}": v for k, v in _metrics(train).items()},
            **{f"oos_{k}": v for k, v in _metrics(oos).items()},
        })

    result = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
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
