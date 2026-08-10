from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from intraday_engine.backtest.engine import backtest_signals
from intraday_engine.strategy.point_in_time import generate_signals


def _floats(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test the rule backtest across score and slippage assumptions")
    parser.add_argument("path", help="CSV file or directory containing *_30d.csv files")
    parser.add_argument("--min-scores", type=_floats, default=[55.0, 60.0, 65.0, 70.0])
    parser.add_argument("--slippage-points", type=_floats, default=[0.0, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--max-holding-bars", type=int, default=30)
    parser.add_argument("--output", default="/tmp/backtest_stress.csv")
    args = parser.parse_args()

    path = Path(args.path)
    files = sorted(path.glob("*_30d.csv")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No *_30d.csv files found in {path}")

    combinations_per_file = len(args.min_scores) * len(args.slippage_points)
    total_runs = len(files) * combinations_per_file
    completed_runs = 0
    started_at = time.perf_counter()

    print(
        f"Stress backtest: {len(files)} symbol files x {len(args.min_scores)} scores x "
        f"{len(args.slippage_points)} slippage values = {total_runs} runs",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for file_number, file in enumerate(files, start=1):
        symbol = file.stem.removesuffix("_30d").upper()
        bars = pd.read_csv(file, parse_dates=["timestamp"])
        print(
            f"[{file_number}/{len(files)}] {symbol}: loaded {len(bars):,} bars",
            flush=True,
        )

        for min_score in args.min_scores:
            # Feature generation is the expensive part. Generate the causal
            # signals once for this score and reuse them across slippage runs.
            signals = generate_signals(
                bars.sort_values("timestamp").reset_index(drop=True),
                symbol=symbol,
                min_score=min_score,
            )
            print(
                f"  score={min_score:g}: {len(signals)} signals; "
                f"running {len(args.slippage_points)} slippage cases",
                flush=True,
            )

            for slippage in args.slippage_points:
                result = backtest_signals(
                    signals,
                    bars,
                    max_holding_bars=args.max_holding_bars,
                    slippage_points=slippage,
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "min_score": min_score,
                        "slippage_points": slippage,
                        "trades": result.total_trades,
                        "wins": result.wins,
                        "win_rate": result.win_rate,
                        "profit_factor": result.profit_factor,
                        "net_points": result.net_points,
                        "max_drawdown_points": result.max_drawdown_points,
                        "expectancy_r": result.expectancy_r,
                    }
                )
                completed_runs += 1
                elapsed = time.perf_counter() - started_at
                rate = completed_runs / elapsed if elapsed > 0 else 0.0
                remaining = (total_runs - completed_runs) / rate if rate > 0 else 0.0
                print(
                    f"    [{completed_runs}/{total_runs}] slip={slippage:g} "
                    f"expectancy={result.expectancy_r:.4f} "
                    f"ETA={remaining:.0f}s",
                    flush=True,
                )

    frame = pd.DataFrame(rows).sort_values(
        ["symbol", "expectancy_r", "profit_factor", "net_points"],
        ascending=[True, False, False, False],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print("===== ROBUSTNESS TOP RESULTS =====")
    for symbol, group in frame.groupby("symbol", sort=True):
        print(f"\n{symbol}")
        print(group.head(5).to_string(index=False))
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
