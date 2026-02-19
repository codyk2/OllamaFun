"""Tests for BacktestResults."""

from datetime import UTC, datetime, timedelta

import pytest

from src.backtesting.results import BacktestResults
from src.core.models import Direction, Trade, TradeStatus


def _make_trade(pnl: float, offset_min: int = 0) -> Trade:
    entry_time = datetime(2024, 1, 15, 9, 30, tzinfo=UTC) + timedelta(minutes=offset_min)
    return Trade(
        strategy="mean_reversion",
        direction=Direction.LONG,
        entry_price=5000.0,
        exit_price=5000.0 + pnl,
        stop_loss=4996.0,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(minutes=10),
        status=TradeStatus.CLOSED,
        pnl_dollars=pnl,
        pnl_ticks=pnl / 1.25,
    )


class TestBacktestResults:
    def test_empty_results(self):
        results = BacktestResults()
        metrics = results.compute_metrics()
        assert metrics.total_trades == 0
        assert results.ending_equity == 10000.0

    def test_compute_metrics(self):
        results = BacktestResults(
            strategy_name="test",
            starting_equity=10000.0,
            trades=[
                _make_trade(50.0, 0),
                _make_trade(-20.0, 20),
                _make_trade(30.0, 40),
            ],
        )
        metrics = results.compute_metrics()
        assert metrics.total_trades == 3
        assert metrics.net_pnl == pytest.approx(60.0)
        assert results.ending_equity == pytest.approx(10060.0)

    def test_summary_string(self):
        results = BacktestResults(
            strategy_name="mean_reversion",
            start_date="2024-01-15",
            end_date="2024-01-16",
            starting_equity=10000.0,
            bars_processed=500,
            signals_generated=10,
            signals_rejected=3,
            trades=[_make_trade(50.0)],
        )
        summary = results.summary()
        assert "mean_reversion" in summary
        assert "500" in summary
        assert "$" in summary

    def test_equity_curve_built(self):
        results = BacktestResults(
            trades=[_make_trade(50.0, 0), _make_trade(-20.0, 20)],
            starting_equity=10000.0,
        )
        results.compute_metrics()
        assert len(results.equity_curve) == 2
