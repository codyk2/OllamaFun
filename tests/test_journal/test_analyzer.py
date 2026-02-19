"""Tests for TradeAnalyzer."""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.models import Direction, Trade, TradeStatus
from src.journal.analyzer import TradeAnalyzer


def _make_trade(
    pnl_dollars: float,
    direction: Direction = Direction.LONG,
    entry_offset_min: int = 0,
    duration_min: int = 15,
    rr_actual: float | None = None,
) -> Trade:
    entry_time = datetime(2024, 1, 15, 9, 30, tzinfo=UTC) + timedelta(minutes=entry_offset_min)
    exit_time = entry_time + timedelta(minutes=duration_min)
    entry_price = 5000.0
    if direction == Direction.LONG:
        exit_price = entry_price + (pnl_dollars / 1.25) * 0.25  # Back-calc from pnl
    else:
        exit_price = entry_price - (pnl_dollars / 1.25) * 0.25

    trade = Trade(
        strategy="mean_reversion",
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=entry_price - 4 if direction == Direction.LONG else entry_price + 4,
        entry_time=entry_time,
        exit_time=exit_time,
        status=TradeStatus.CLOSED,
        pnl_dollars=pnl_dollars,
        pnl_ticks=pnl_dollars / 1.25,
        risk_reward_actual=rr_actual,
    )
    return trade


class TestTradeAnalyzer:
    def setup_method(self):
        self.analyzer = TradeAnalyzer()

    def test_empty_trades(self):
        metrics = self.analyzer.analyze([])
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.net_pnl == 0.0

    def test_all_winners(self):
        trades = [
            _make_trade(50.0, entry_offset_min=0),
            _make_trade(30.0, entry_offset_min=20),
            _make_trade(20.0, entry_offset_min=40),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.total_trades == 3
        assert metrics.winners == 3
        assert metrics.losers == 0
        assert metrics.win_rate == 1.0
        assert metrics.net_pnl == pytest.approx(100.0)
        assert metrics.gross_profit == pytest.approx(100.0)
        assert metrics.gross_loss == 0.0

    def test_all_losers(self):
        trades = [
            _make_trade(-30.0, entry_offset_min=0),
            _make_trade(-20.0, entry_offset_min=20),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.total_trades == 2
        assert metrics.winners == 0
        assert metrics.losers == 2
        assert metrics.win_rate == 0.0
        assert metrics.net_pnl == pytest.approx(-50.0)

    def test_mixed_trades(self):
        trades = [
            _make_trade(50.0, entry_offset_min=0),
            _make_trade(-20.0, entry_offset_min=20),
            _make_trade(30.0, entry_offset_min=40),
            _make_trade(-10.0, entry_offset_min=60),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.total_trades == 4
        assert metrics.winners == 2
        assert metrics.losers == 2
        assert metrics.win_rate == pytest.approx(0.5)
        assert metrics.net_pnl == pytest.approx(50.0)
        assert metrics.gross_profit == pytest.approx(80.0)
        assert metrics.gross_loss == pytest.approx(-30.0)
        assert metrics.profit_factor == pytest.approx(80.0 / 30.0)

    def test_max_winner_and_loser(self):
        trades = [
            _make_trade(100.0, entry_offset_min=0),
            _make_trade(20.0, entry_offset_min=20),
            _make_trade(-50.0, entry_offset_min=40),
            _make_trade(-10.0, entry_offset_min=60),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.max_winner == pytest.approx(100.0)
        assert metrics.max_loser == pytest.approx(-50.0)

    def test_profit_factor_no_losses(self):
        trades = [_make_trade(50.0)]
        metrics = self.analyzer.analyze(trades)
        assert metrics.profit_factor == float("inf")

    def test_drawdown(self):
        trades = [
            _make_trade(50.0, entry_offset_min=0),
            _make_trade(-30.0, entry_offset_min=20),
            _make_trade(-20.0, entry_offset_min=40),
            _make_trade(60.0, entry_offset_min=60),
        ]
        metrics = self.analyzer.analyze(trades)
        # Peak at 50, then drops to 50-30-20=0, so max dd = 50
        assert metrics.max_drawdown == pytest.approx(50.0)

    def test_sharpe_ratio_single_trade(self):
        trades = [_make_trade(10.0)]
        metrics = self.analyzer.analyze(trades)
        assert metrics.sharpe_ratio is None  # Need at least 2 trades

    def test_sharpe_ratio_with_variation(self):
        trades = [
            _make_trade(10.0, entry_offset_min=0),
            _make_trade(-5.0, entry_offset_min=20),
            _make_trade(15.0, entry_offset_min=40),
            _make_trade(-3.0, entry_offset_min=60),
            _make_trade(8.0, entry_offset_min=80),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.sharpe_ratio is not None
        assert metrics.sharpe_ratio > 0  # Net positive P&L

    def test_sharpe_none_for_constant_pnl(self):
        trades = [
            _make_trade(10.0, entry_offset_min=i * 20)
            for i in range(5)
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.sharpe_ratio is None  # Zero std

    def test_winning_streak(self):
        trades = [
            _make_trade(10.0, entry_offset_min=0),
            _make_trade(20.0, entry_offset_min=20),
            _make_trade(15.0, entry_offset_min=40),
            _make_trade(-5.0, entry_offset_min=60),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.winning_streak == 3
        assert metrics.losing_streak == 1
        assert metrics.current_streak == -1

    def test_losing_streak(self):
        trades = [
            _make_trade(10.0, entry_offset_min=0),
            _make_trade(-5.0, entry_offset_min=20),
            _make_trade(-8.0, entry_offset_min=40),
            _make_trade(-3.0, entry_offset_min=60),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.losing_streak == 3
        assert metrics.current_streak == -3

    def test_avg_duration(self):
        trades = [
            _make_trade(10.0, duration_min=10, entry_offset_min=0),
            _make_trade(-5.0, duration_min=20, entry_offset_min=30),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.avg_trade_duration_minutes == pytest.approx(15.0)

    def test_open_trades_excluded(self):
        """Open trades should not be included in analysis."""
        open_trade = Trade(
            strategy="test",
            direction=Direction.LONG,
            entry_price=5000.0,
            stop_loss=4996.0,
            entry_time=datetime.now(UTC),
            status=TradeStatus.OPEN,
        )
        closed = _make_trade(50.0)
        metrics = self.analyzer.analyze([open_trade, closed])
        assert metrics.total_trades == 1

    def test_breakeven_trades(self):
        trades = [_make_trade(0.0)]
        metrics = self.analyzer.analyze(trades)
        assert metrics.breakeven == 1
        assert metrics.winners == 0
        assert metrics.losers == 0

    def test_risk_reward_average(self):
        trades = [
            _make_trade(50.0, rr_actual=2.5, entry_offset_min=0),
            _make_trade(-20.0, rr_actual=-1.0, entry_offset_min=20),
        ]
        metrics = self.analyzer.analyze(trades)
        assert metrics.avg_risk_reward == pytest.approx(0.75)


class TestEquityCurve:
    def setup_method(self):
        self.analyzer = TradeAnalyzer()

    def test_equity_curve_basic(self):
        trades = [
            _make_trade(50.0, entry_offset_min=0),
            _make_trade(-20.0, entry_offset_min=20),
        ]
        curve = self.analyzer.compute_equity_curve(trades, 10000.0)
        assert len(curve) == 2
        assert curve[0][1] == pytest.approx(10050.0)
        assert curve[1][1] == pytest.approx(10030.0)

    def test_empty_trades(self):
        curve = self.analyzer.compute_equity_curve([], 10000.0)
        assert curve == []


class TestDrawdownSeries:
    def setup_method(self):
        self.analyzer = TradeAnalyzer()

    def test_drawdown_series(self):
        curve = [(1.0, 10000.0), (2.0, 10050.0), (3.0, 10020.0), (4.0, 10060.0)]
        dd = self.analyzer.compute_drawdown_series(curve)
        assert len(dd) == 4
        assert dd[0][1] == pytest.approx(0.0)
        assert dd[1][1] == pytest.approx(0.0)
        assert dd[2][1] == pytest.approx(30.0)
        assert dd[3][1] == pytest.approx(0.0)

    def test_empty_curve(self):
        assert self.analyzer.compute_drawdown_series([]) == []
