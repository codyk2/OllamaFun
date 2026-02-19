"""Tests for walk-forward analysis."""

import random
from datetime import UTC, datetime, timedelta

import pytest

from src.backtesting.walk_forward import WalkForwardAnalyzer, WalkForwardReport
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


class TestWalkForwardReport:
    def test_empty_report(self):
        report = WalkForwardReport()
        assert len(report.folds) == 0
        assert report.total_oos_trades == 0

    def test_summary_string(self):
        report = WalkForwardReport()
        s = report.summary()
        assert "Walk-Forward" in s
        assert "0 folds" in s


class TestWalkForwardAnalyzer:
    def test_creates_analyzer(self):
        bars = _make_bars(500)
        analyzer = WalkForwardAnalyzer(
            bars=bars, is_bars=200, oos_bars=100, trials_per_fold=3,
        )
        assert analyzer.is_bars == 200
        assert analyzer.oos_bars == 100

    def test_insufficient_bars_returns_empty(self):
        """Not enough bars for even one fold should return empty report."""
        bars = _make_bars(100)
        analyzer = WalkForwardAnalyzer(
            bars=bars, is_bars=200, oos_bars=100, trials_per_fold=3,
        )
        report = analyzer.run()
        assert len(report.folds) == 0

    def test_single_fold_runs(self):
        """Enough bars for exactly one fold."""
        bars = _make_bars(600, volatility=3.0)
        analyzer = WalkForwardAnalyzer(
            bars=bars, is_bars=300, oos_bars=200, trials_per_fold=3,
        )
        report = analyzer.run()
        assert len(report.folds) >= 1
        assert report.folds[0].is_bars == 300
        assert report.folds[0].oos_bars == 200

    def test_multiple_folds(self):
        """Enough bars for multiple folds."""
        bars = _make_bars(1000, volatility=3.0)
        analyzer = WalkForwardAnalyzer(
            bars=bars, is_bars=300, oos_bars=200, trials_per_fold=3,
        )
        report = analyzer.run()
        # With 1000 bars, IS=300, OOS=200, fold_size=500
        # First fold: 0-500, second fold: 200-700, third: 400-900, fourth: 600-1000
        assert len(report.folds) >= 2
