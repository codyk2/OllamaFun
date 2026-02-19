"""Optuna-based parameter optimizer for the mean reversion strategy.

Searches over 8 tunable parameters using TPE sampler with a composite
objective: Sharpe ratio × trade count penalty × drawdown penalty.
"""

from __future__ import annotations

import optuna

from src.backtesting.engine import BacktestEngine
from src.backtesting.results import BacktestResults
from src.core.logging import get_logger
from src.core.models import Bar
from src.strategies.base import StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy

logger = get_logger("optimizer")

# Parameter search space
PARAM_SPACE = {
    "bb_touch_threshold": (0.0, 1.0),
    "rsi_oversold": (25.0, 40.0),
    "rsi_overbought": (60.0, 75.0),
    "atr_stop_multiple": (1.5, 3.0),
    "risk_reward_target": (0.8, 2.0),
    "min_atr": (0.3, 1.5),
    "rsi_extreme_oversold": (15.0, 30.0),
    "rsi_extreme_overbought": (70.0, 85.0),
}

# Minimum trades to consider a valid result
MIN_TRADES_FOR_VALID = 10


def composite_objective(results: BacktestResults) -> float:
    """Compute optimization objective.

    Maximizes: Sharpe × trade_count_factor × drawdown_factor

    - Sharpe ratio: primary performance metric
    - Trade count penalty: penalizes too few trades (overfitting risk)
    - Drawdown penalty: penalizes excessive drawdowns
    """
    if results.metrics is None:
        return -10.0

    m = results.metrics
    if m.total_trades < MIN_TRADES_FOR_VALID:
        return -10.0

    sharpe = m.sharpe_ratio or 0.0

    # Trade count factor: ramp from 0 at 10 trades to 1 at 30+ trades
    trade_factor = min(1.0, (m.total_trades - MIN_TRADES_FOR_VALID) / 20.0)

    # Drawdown penalty: penalize drawdowns > 5% of starting equity
    if results.starting_equity > 0:
        dd_pct = m.max_drawdown / results.starting_equity
        dd_factor = max(0.0, 1.0 - dd_pct * 5)  # Linear penalty
    else:
        dd_factor = 1.0

    score = sharpe * trade_factor * dd_factor

    return score


class OptunaOptimizer:
    """Parameter optimizer using Optuna's TPE sampler."""

    def __init__(
        self,
        bars: list[Bar],
        starting_equity: float = 10000.0,
        n_trials: int = 300,
        risk_config: dict | None = None,
    ) -> None:
        self.bars = bars
        self.starting_equity = starting_equity
        self.n_trials = n_trials
        self.risk_config = risk_config

    def _objective(self, trial: optuna.Trial) -> float:
        """Single trial: suggest params, run backtest, return score."""
        params = {}
        for name, (low, high) in PARAM_SPACE.items():
            params[name] = trial.suggest_float(name, low, high)

        # Ensure rsi_oversold < rsi_overbought
        if params["rsi_oversold"] >= params["rsi_overbought"]:
            return -10.0

        # Ensure extreme thresholds are beyond standard thresholds
        if params["rsi_extreme_oversold"] >= params["rsi_oversold"]:
            return -10.0
        if params["rsi_extreme_overbought"] <= params["rsi_overbought"]:
            return -10.0

        config = StrategyConfig(name="mean_reversion", params=params)
        try:
            strategy = MeanReversionStrategy(config=config)
        except (AssertionError, ValueError):
            return -10.0

        engine = BacktestEngine(
            strategies=[strategy],
            starting_equity=self.starting_equity,
            risk_config=self.risk_config,
        )
        results = engine.run(self.bars)
        return composite_objective(results)

    def optimize(self) -> tuple[dict, float]:
        """Run optimization and return best params + score."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        logger.info(
            "optimization_complete",
            best_score=study.best_value,
            n_trials=self.n_trials,
            best_params=study.best_params,
        )

        return study.best_params, study.best_value
