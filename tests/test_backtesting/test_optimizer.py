"""Tests for Optuna parameter optimizer."""

import random
from datetime import UTC, datetime, timedelta

import pytest

from src.backtesting.optimizer import (
    MIN_TRADES_FOR_VALID,
    PARAM_SPACE,
    OptunaOptimizer,
    composite_objective,
)
from src.backtesting.results import BacktestResults
from src.core.models import Bar


def _make_bars(count=500, start_price=5000.0, volatility=2.5, seed=42) -> list[Bar]:
    random.seed(seed)
    bars = []
    price = start_price
    base_time = datetime(2024, 1, 15, 9, 30, tzinfo=UTC)
    for i in range(count):
        change = random.gauss(0, volatility)
        price += change
        high = price + abs(random.gauss(0, 1))
        low = price - abs(random.gauss(0, 1))
        bars.append(Bar(
            timestamp=base_time + timedelta(minutes=i),
            open=price - change / 2,
            high=max(high, price),
            low=min(low, price),
            close=price,
            volume=random.randint(100, 5000),
        ))
    return bars


class TestCompositeObjective:
    def test_no_metrics_returns_negative(self):
        results = BacktestResults(
            strategy_name="test", starting_equity=10000, ending_equity=10000
        )
        score = composite_objective(results)
        assert score == -10.0

    def test_too_few_trades_returns_negative(self):
        results = BacktestResults(
            strategy_name="test", starting_equity=10000, ending_equity=10000
        )
        results.compute_metrics()
        score = composite_objective(results)
        assert score == -10.0

    def test_param_space_has_expected_keys(self):
        expected = {
            "bb_touch_threshold", "rsi_oversold", "rsi_overbought",
            "atr_stop_multiple", "risk_reward_target", "min_atr",
            "rsi_extreme_oversold", "rsi_extreme_overbought",
        }
        assert set(PARAM_SPACE.keys()) == expected

    def test_param_ranges_valid(self):
        for name, (low, high) in PARAM_SPACE.items():
            assert low < high, f"{name} has invalid range"


class TestOptunaOptimizer:
    def test_creates_optimizer(self):
        bars = _make_bars(100)
        opt = OptunaOptimizer(bars=bars, n_trials=5)
        assert opt.n_trials == 5

    def test_optimize_runs_without_error(self):
        """Smoke test: run a few trials."""
        bars = _make_bars(300, volatility=3.0)
        opt = OptunaOptimizer(bars=bars, n_trials=5, starting_equity=10000.0)
        best_params, best_score = opt.optimize()
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert len(best_params) > 0

    def test_best_params_have_expected_keys(self):
        bars = _make_bars(300, volatility=3.0)
        opt = OptunaOptimizer(bars=bars, n_trials=5)
        best_params, _ = opt.optimize()
        for key in PARAM_SPACE:
            assert key in best_params
