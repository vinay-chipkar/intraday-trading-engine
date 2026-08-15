from __future__ import annotations

import argparse

import pandas as pd

from intraday_engine.backtest.costs import (
    CostModel,
    apply_cost_model,
    backtest_trades_to_cost_rows,
    build_cost_comparison_report,
)
from intraday_engine.backtest.pipeline import run_rule_backtest

COST_PRESETS = {
    "zero": CostModel.zero_cost,
    "realistic": CostModel.realistic,
    "conservative": CostModel.conservative_stress,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the rule-based intraday engine")
    parser.add_argument("csv", help="CSV containing timestamp, open, high, low, close and optional volume")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--market-score", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--max-holding-bars", type=int, default=30)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument(
        "--cost-model",
        choices=sorted(COST_PRESETS) + ["all"],
        default=None,
        help="Report gross-vs-net P&L under a transaction-cost preset (research-only; "
        "never changes the simulated trades themselves). 'all' prints zero/realistic/conservative side by side.",
    )
    args = parser.parse_args()

    bars = pd.read_csv(args.csv, parse_dates=["timestamp"])
    result = run_rule_backtest(
        bars,
        symbol=args.symbol,
        market_score=args.market_score,
        min_score=args.min_score,
        max_holding_bars=args.max_holding_bars,
        slippage_points=args.slippage_points,
    )

    print(f"trades={result.total_trades}")
    print(f"wins={result.wins}")
    print(f"losses={result.losses}")
    print(f"timeouts={result.timeouts}")
    print(f"win_rate={result.win_rate:.4f}")
    print(f"profit_factor={result.profit_factor}")
    print(f"net_points={result.net_points:.4f}")
    print(f"max_drawdown_points={result.max_drawdown_points:.4f}")
    print(f"expectancy_r={result.expectancy_r:.4f}")

    if args.cost_model == "all":
        rows = backtest_trades_to_cost_rows(result.trades)
        comparison = build_cost_comparison_report(rows)
        print("\n--- cost comparison: gross vs. realistic-net vs. conservative-net ---")
        print(f"trades={comparison.trades}")
        print(f"gross_pnl_points={comparison.gross_pnl_points:.4f}")
        print(f"realistic_net_pnl_points={comparison.realistic_net_pnl_points:.4f}")
        print(f"conservative_net_pnl_points={comparison.conservative_net_pnl_points:.4f}")
        print(f"gross_expectancy_r={comparison.gross_expectancy_r:.4f}")
        print(f"realistic_net_expectancy_r={comparison.realistic_net_expectancy_r:.4f}")
        print(f"conservative_net_expectancy_r={comparison.conservative_net_expectancy_r:.4f}")
        print(f"gross_profit_factor={comparison.gross_profit_factor}")
        print(f"realistic_net_profit_factor={comparison.realistic_net_profit_factor}")
        print(f"conservative_net_profit_factor={comparison.conservative_net_profit_factor}")
    elif args.cost_model:
        rows = backtest_trades_to_cost_rows(result.trades)
        report = apply_cost_model(rows, COST_PRESETS[args.cost_model]())
        print(f"\n--- cost model: {args.cost_model} ---")
        print(f"gross_pnl_points={report.gross_pnl_points:.4f}")
        print(f"slippage_points={report.slippage_points:.4f}")
        print(f"transaction_cost_points={report.transaction_cost_points:.4f}")
        print(f"net_pnl_points={report.net_pnl_points:.4f}")
        print(f"gross_expectancy_r={report.gross_expectancy_r:.4f}")
        print(f"net_expectancy_r={report.net_expectancy_r:.4f}")
        print(f"gross_profit_factor={report.gross_profit_factor}")
        print(f"net_profit_factor={report.net_profit_factor}")


if __name__ == "__main__":
    main()
