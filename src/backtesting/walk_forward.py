"""Walk-forward analysis: rolling optimization + out-of-sample validation.

Splits data into rolling windows:
  - In-sample (IS): optimize parameters
  - Out-of-sample (OOS): test on unseen data

Default: 6-month IS / 2-month OOS, rolling forward by OOS window size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backtesting.engine import BacktestEngine
from src.backtesting.optimizer import OptunaOptimizer
from src.backtesting.results import BacktestResults
from src.core.logging import get_logger
from src.core.models import Bar
from src.strategies.base import StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy

logger = get_logger("walk_forward")


@dataclass
class WalkForwardFold:
    """Results for a single walk-forward fold."""

    fold_number: int
    is_bars: int
    oos_bars: int
    best_params: dict
    is_score: float
    oos_results: BacktestResults | None = None
    oos_score: float = 0.0


@dataclass
class WalkForwardReport:
    """Aggregate results across all walk-forward folds."""

    folds: list[WalkForwardFold] = field(default_factory=list)
    total_oos_trades: int = 0
    total_oos_pnl: float = 0.0
    avg_is_score: float = 0.0
    avg_oos_score: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Analysis: {len(self.folds)} folds",
            f"  Total OOS Trades: {self.total_oos_trades}",
            f"  Total OOS P&L:    ${self.total_oos_pnl:+.2f}",
            f"  Avg IS Score:     {self.avg_is_score:.3f}",
            f"  Avg OOS Score:    {self.avg_oos_score:.3f}",
        ]
        for fold in self.folds:
            lines.append(
                f"  Fold {fold.fold_number}: IS={fold.is_score:.3f} "
                f"OOS={fold.oos_score:.3f} trades={fold.oos_results.metrics.total_trades if fold.oos_results and fold.oos_results.metrics else 0}"
            )
        return "\n".join(lines)


class WalkForwardAnalyzer:
    """Rolling walk-forward optimization and validation."""

    def __init__(
        self,
        bars: list[Bar],
        is_bars: int = 7800,    # ~6 months of 1-min bars (65 bars/day × 120 days)
        oos_bars: int = 2600,   # ~2 months of 1-min bars
        trials_per_fold: int = 100,
        starting_equity: float = 10000.0,
        risk_config: dict | None = None,
    ) -> None:
        self.bars = bars
        self.is_bars = is_bars
        self.oos_bars = oos_bars
        self.trials_per_fold = trials_per_fold
        self.starting_equity = starting_equity
        self.risk_config = risk_config

    def run(self) -> WalkForwardReport:
        """Execute walk-forward analysis across all folds."""
        report = WalkForwardReport()
        total_bars = len(self.bars)
        fold_size = self.is_bars + self.oos_bars
        fold_num = 0

        start = 0
        while start + fold_size <= total_bars:
            fold_num += 1
            is_end = start + self.is_bars
            oos_end = is_end + self.oos_bars

            is_data = self.bars[start:is_end]
            oos_data = self.bars[is_end:oos_end]

            logger.info(
                "walk_forward_fold",
                fold=fold_num,
                is_bars=len(is_data),
                oos_bars=len(oos_data),
            )

            # Optimize on in-sample
            optimizer = OptunaOptimizer(
                bars=is_data,
                starting_equity=self.starting_equity,
                n_trials=self.trials_per_fold,
                risk_config=self.risk_config,
            )
            best_params, is_score = optimizer.optimize()

            # Validate on out-of-sample
            config = StrategyConfig(name="mean_reversion", params=best_params)
            try:
                strategy = MeanReversionStrategy(config=config)
            except (AssertionError, ValueError):
                fold = WalkForwardFold(
                    fold_number=fold_num,
                    is_bars=len(is_data),
                    oos_bars=len(oos_data),
                    best_params=best_params,
                    is_score=is_score,
                )
                report.folds.append(fold)
                start += self.oos_bars
                continue

            engine = BacktestEngine(
                strategies=[strategy],
                starting_equity=self.starting_equity,
                risk_config=self.risk_config,
            )
            oos_results = engine.run(oos_data)

            from src.backtesting.optimizer import composite_objective
            oos_score = composite_objective(oos_results)

            fold = WalkForwardFold(
                fold_number=fold_num,
                is_bars=len(is_data),
                oos_bars=len(oos_data),
                best_params=best_params,
                is_score=is_score,
                oos_results=oos_results,
                oos_score=oos_score,
            )
            report.folds.append(fold)

            if oos_results.metrics:
                report.total_oos_trades += oos_results.metrics.total_trades
                report.total_oos_pnl += oos_results.metrics.net_pnl

            start += self.oos_bars  # Roll forward by OOS window

        if report.folds:
            report.avg_is_score = sum(f.is_score for f in report.folds) / len(report.folds)
            report.avg_oos_score = sum(f.oos_score for f in report.folds) / len(report.folds)

        logger.info(
            "walk_forward_complete",
            folds=len(report.folds),
            oos_trades=report.total_oos_trades,
            oos_pnl=report.total_oos_pnl,
        )

        return report
