from __future__ import annotations

import argparse

import pandas as pd

from intraday_engine.backtest.pipeline import run_rule_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the rule-based intraday engine")
    parser.add_argument("csv", help="CSV containing timestamp, open, high, low, close and optional volume")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--market-score", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--max-holding-bars", type=int, default=30)
    parser.add_argument("--slippage-points", type=float, default=0.0)
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


if __name__ == "__main__":
    main()
