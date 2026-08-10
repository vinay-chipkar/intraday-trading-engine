from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from intraday_engine.backtest.diagnostics import summarize_diagnostics, trade_diagnostics
from intraday_engine.backtest.pipeline import run_rule_backtest
from intraday_engine.strategy.point_in_time import generate_signals


def _run(
    path: Path,
    symbol: str,
    *,
    min_score: float,
    max_holding_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars = pd.read_csv(path, parse_dates=["timestamp"])
    signals = generate_signals(bars, symbol=symbol, min_score=min_score)
    result = run_rule_backtest(
        bars,
        symbol=symbol,
        min_score=min_score,
        max_holding_bars=max_holding_bars,
    )
    trades = trade_diagnostics(result, signals)
    by_side, by_score, by_reason, by_excursion = summarize_diagnostics(trades)
    return trades, by_side, by_score, by_reason, by_excursion


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose rule-based backtest trades")
    parser.add_argument("path", help="CSV file or directory containing *_30d.csv files")
    parser.add_argument("--symbol", help="Symbol when path is a single CSV")
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--max-holding-bars", type=int, default=30)
    parser.add_argument("--output-dir", default="/tmp/backtest_analysis")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        files = sorted(path.glob("*_30d.csv"))
    else:
        files = [path]

    if not files:
        raise SystemExit(f"No *_30d.csv files found in {path}")
    if len(files) == 1 and not args.symbol:
        symbol = files[0].stem.removesuffix("_30d")
        items = [(files[0], symbol)]
    elif args.symbol:
        items = [(files[0], args.symbol.upper())]
    else:
        items = [(file, file.stem.removesuffix("_30d")) for file in files]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trades: list[pd.DataFrame] = []
    all_sides: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_reasons: list[pd.DataFrame] = []
    all_excursions: list[pd.DataFrame] = []

    for file, symbol in items:
        trades, by_side, by_score, by_reason, by_excursion = _run(
            file,
            symbol,
            min_score=args.min_score,
            max_holding_bars=args.max_holding_bars,
        )
        all_trades.append(trades)
        all_sides.append(by_side)
        all_scores.append(by_score)
        all_reasons.append(by_reason)
        all_excursions.append(by_excursion)

        print(f"\n===== {symbol} =====")
        if by_side.empty:
            print("No completed trades")
        else:
            print("\nBy side:")
            print(by_side.to_string(index=False))
        if not by_score.empty:
            print("\nBy score magnitude and side:")
            print(by_score.to_string(index=False))
        if not by_excursion.empty:
            print("\nBy excursion:")
            print(by_excursion.to_string(index=False))
        if not by_reason.empty:
            print("\nBy reason:")
            print(by_reason.to_string(index=False))

    pd.concat(all_trades, ignore_index=True).to_csv(output_dir / "trades.csv", index=False)
    pd.concat(all_sides, ignore_index=True).to_csv(output_dir / "by_side.csv", index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(output_dir / "by_score.csv", index=False)
    pd.concat(all_excursions, ignore_index=True).to_csv(output_dir / "by_excursion.csv", index=False)
    if any(not frame.empty for frame in all_reasons):
        pd.concat(
            [frame for frame in all_reasons if not frame.empty], ignore_index=True
        ).to_csv(output_dir / "by_reason.csv", index=False)

    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main()
