from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from intraday_engine.backtest.pipeline import run_rule_backtest


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

    rows: list[dict[str, object]] = []
    for file in files:
        symbol = file.stem.removesuffix("_30d").upper()
        bars = pd.read_csv(file, parse_dates=["timestamp"])
        for min_score in args.min_scores:
            for slippage in args.slippage_points:
                result = run_rule_backtest(
                    bars,
                    symbol=symbol,
                    min_score=min_score,
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
