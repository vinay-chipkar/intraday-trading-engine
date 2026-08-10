from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from intraday_engine.backtest.research import (
    FilterSpec,
    default_filter_specs,
    evaluate_filter,
)


def _metric_row(
    symbol: str,
    side: str,
    spec: FilterSpec,
    result,
    split: str,
) -> dict[str, object]:
    trades = [trade for trade in result.trades if trade.side == side]
    if not trades:
        return {
            "symbol": symbol,
            "side": side,
            "filter": spec.name,
            "split": split,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "net_points": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_points": 0.0,
        }

    gross_profit = sum(t.pnl_points for t in trades if t.pnl_points > 0)
    gross_loss = -sum(t.pnl_points for t in trades if t.pnl_points < 0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for trade in sorted(trades, key=lambda t: t.exit_time):
        equity += trade.pnl_points
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    wins = sum(t.pnl_points > 0 for t in trades)
    return {
        "symbol": symbol,
        "side": side,
        "filter": spec.name,
        "split": split,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "net_points": sum(t.pnl_points for t in trades),
        "expectancy_r": sum(t.r_multiple for t in trades) / len(trades),
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
        "max_drawdown_points": max_dd,
    }


def _evaluate_file(
    path: Path,
    *,
    min_score: float,
    max_holding_bars: int,
    train_fraction: float,
) -> pd.DataFrame:
    symbol = path.stem.removesuffix("_30d").upper()
    bars = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(bars) < 100:
        raise ValueError(f"{symbol}: not enough bars for walk-forward research ({len(bars)})")

    split_index = max(1, min(len(bars) - 1, int(len(bars) * train_fraction)))
    split_time = pd.Timestamp(bars.iloc[split_index]["timestamp"])
    train_bars = bars.iloc[:split_index].copy()

    rows: list[dict[str, object]] = []
    for spec in default_filter_specs():
        # The production engine remains unchanged. This research layer only
        # evaluates pre-declared entry filters against the existing signals.
        train_result = evaluate_filter(
            train_bars,
            symbol=symbol,
            spec=spec,
            min_score=min_score,
            max_holding_bars=max_holding_bars,
            end_time=split_time,
        )
        test_result = evaluate_filter(
            bars,
            symbol=symbol,
            spec=spec,
            min_score=min_score,
            max_holding_bars=max_holding_bars,
            start_time=split_time,
        )
        for side in ("LONG", "SHORT"):
            rows.append(_metric_row(symbol, side, spec, train_result, "train"))
            rows.append(_metric_row(symbol, side, spec, test_result, "test"))

    return pd.DataFrame(rows)


def _choose_train_winners(metrics: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    train = metrics[(metrics["split"] == "train") & (metrics["trades"] >= min_trades)].copy()
    if train.empty:
        return pd.DataFrame(columns=["symbol", "side", "filter", "train_trades", "train_expectancy_r", "train_net_points"])
    train = train.sort_values(
        ["symbol", "side", "expectancy_r", "net_points"],
        ascending=[True, True, False, False],
    )
    winners = train.groupby(["symbol", "side"], as_index=False).head(1).copy()
    return winners.rename(
        columns={
            "trades": "train_trades",
            "expectancy_r": "train_expectancy_r",
            "net_points": "train_net_points",
        }
    )[["symbol", "side", "filter", "train_trades", "train_expectancy_r", "train_net_points"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward research of declared signal filters")
    parser.add_argument("path", help="CSV file or directory containing *_30d.csv files")
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--max-holding-bars", type=int, default=30)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--output-dir", default="/tmp/filter_research")
    args = parser.parse_args()

    if not 0.5 <= args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be >= 0.5 and < 1.0")

    path = Path(args.path)
    files = sorted(path.glob("*_30d.csv")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No *_30d.csv files found in {path}")

    metrics = pd.concat(
        [
            _evaluate_file(
                file,
                min_score=args.min_score,
                max_holding_bars=args.max_holding_bars,
                train_fraction=args.train_fraction,
            )
            for file in files
        ],
        ignore_index=True,
    )

    winners = _choose_train_winners(metrics, args.min_trades)
    test = metrics[metrics["split"] == "test"].copy()
    test = test.merge(winners[["symbol", "side", "filter"]], on=["symbol", "side", "filter"], how="inner")
    test = test.sort_values(["symbol", "side"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "all_filters.csv", index=False)
    winners.to_csv(output_dir / "train_winners.csv", index=False)
    test.to_csv(output_dir / "winner_out_of_sample.csv", index=False)

    print("\n===== TRAIN WINNERS =====")
    print(winners.to_string(index=False) if not winners.empty else "No filter met the minimum trade count")
    print("\n===== OUT-OF-SAMPLE RESULTS =====")
    print(test.to_string(index=False) if not test.empty else "No selected filters had out-of-sample results")
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
