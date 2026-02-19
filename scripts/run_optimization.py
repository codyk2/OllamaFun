"""Run parameter optimization with optional walk-forward validation.

Usage:
  Basic:         python -m scripts.run_optimization
  More trials:   python -m scripts.run_optimization --trials 500
  Walk-forward:  python -m scripts.run_optimization --walk-forward
  From CSV:      python -m scripts.run_optimization --csv data/mes_1m.csv
"""

import argparse
import sys
import time

from src.backtesting.optimizer import OptunaOptimizer
from src.core.logging import get_logger, setup_logging
from src.market_data.historical import generate_sample_bars, load_csv_bars

logger = get_logger("run_optimization")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--csv", type=str, help="Path to CSV file with OHLCV bars")
    parser.add_argument("--bars", type=int, default=2000, help="Number of synthetic bars")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting equity")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--volatility", type=float, default=2.5, help="Synthetic bar volatility")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    parser.add_argument("--wf-is-bars", type=int, default=1000, help="Walk-forward IS window")
    parser.add_argument("--wf-oos-bars", type=int, default=300, help="Walk-forward OOS window")
    args = parser.parse_args()

    # Load bars
    if args.csv:
        print(f"Loading bars from {args.csv}...")
        bars = load_csv_bars(args.csv)
        if not bars:
            print("ERROR: No bars loaded from CSV.")
            sys.exit(1)
    else:
        print(f"Generating {args.bars} synthetic bars (volatility={args.volatility})...")
        bars = generate_sample_bars(
            count=args.bars,
            start_price=5950.0,
            volatility=args.volatility,
        )

    print(f"Loaded {len(bars)} bars")
    print()

    if args.walk_forward:
        from src.backtesting.walk_forward import WalkForwardAnalyzer

        print(f"Running walk-forward analysis ({args.trials} trials/fold)...")
        start = time.time()
        analyzer = WalkForwardAnalyzer(
            bars=bars,
            is_bars=args.wf_is_bars,
            oos_bars=args.wf_oos_bars,
            trials_per_fold=args.trials,
            starting_equity=args.equity,
        )
        report = analyzer.run()
        elapsed = time.time() - start

        print()
        print("=" * 60)
        print(report.summary())
        print("=" * 60)
        print(f"\nCompleted in {elapsed:.1f}s")
    else:
        print(f"Running optimization ({args.trials} trials)...")
        start = time.time()
        optimizer = OptunaOptimizer(
            bars=bars,
            starting_equity=args.equity,
            n_trials=args.trials,
        )
        best_params, best_score = optimizer.optimize()
        elapsed = time.time() - start

        print()
        print("=" * 60)
        print(f"BEST SCORE: {best_score:.4f}")
        print("-" * 60)
        print("BEST PARAMETERS:")
        for key, value in sorted(best_params.items()):
            print(f"  {key:30s} = {value:.4f}")
        print("=" * 60)
        print(f"\nCompleted in {elapsed:.1f}s ({args.trials} trials)")

        # Run a final backtest with best params
        from src.backtesting.engine import BacktestEngine
        from src.strategies.base import StrategyConfig
        from src.strategies.mean_reversion import MeanReversionStrategy

        config = StrategyConfig(name="mean_reversion", params=best_params)
        strategy = MeanReversionStrategy(config=config)
        engine = BacktestEngine(strategies=[strategy], starting_equity=args.equity)
        results = engine.run(bars)

        print()
        print(results.summary())
        if results.metrics:
            m = results.metrics
            print(f"  Trades: {m.total_trades}  Win Rate: {m.win_rate:.1%}")
            print(f"  Net P&L: ${m.net_pnl:+.2f}  Sharpe: {m.sharpe_ratio:.2f}" if m.sharpe_ratio else f"  Net P&L: ${m.net_pnl:+.2f}")


if __name__ == "__main__":
    main()
