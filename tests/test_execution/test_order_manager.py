"""Tests for OrderManager."""

import random
from datetime import UTC, datetime

import pytest

from src.core.models import (
    Bar,
    Direction,
    Position,
    RiskDecision,
    RiskResult,
    Signal,
    Trade,
    TradeStatus,
)
from src.execution.order_manager import OrderManager
from src.execution.paper_executor import PaperExecutor
from src.journal.recorder import TradeRecorder
from src.risk.manager import RiskManager


def _make_signal(direction=Direction.LONG, price=5000.0):
    return Signal(
        strategy="mean_reversion",
        direction=direction,
        confidence=0.7,
        entry_price=price,
        stop_loss=price - 4 if direction == Direction.LONG else price + 4,
        take_profit=price + 8 if direction == Direction.LONG else price - 8,
    )


def _make_risk_result(direction=Direction.LONG, price=5000.0, qty=1):
    return RiskResult(
        decision=RiskDecision.APPROVED,
        position_size=qty,
        reason="approved",
        signal=_make_signal(direction, price),
    )


def _make_rejected_result():
    return RiskResult(
        decision=RiskDecision.REJECTED,
        position_size=0,
        reason="rejected",
        signal=_make_signal(),
    )


def _make_bar(close=5000.0):
    return Bar(
        timestamp=datetime.now(UTC),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1000,
    )


@pytest.fixture
def executor():
    return PaperExecutor(
        paper_mode=True,
        slippage_ticks_mean=0,
        slippage_ticks_std=0,
    )


@pytest.fixture
def risk_manager():
    return RiskManager(account_equity=10000.0)


@pytest.fixture
def order_manager(executor, risk_manager):
    return OrderManager(executor=executor, risk_manager=risk_manager)


class TestProcessSignals:
    def test_approved_signal_opens_position(self, order_manager):
        results = [_make_risk_result(Direction.LONG, 5000.0)]
        trades = order_manager.process_signals(results)
        assert len(trades) == 1
        assert trades[0].status == TradeStatus.OPEN
        assert len(order_manager.open_positions) == 1

    def test_rejected_signal_skipped(self, order_manager):
        results = [_make_rejected_result()]
        trades = order_manager.process_signals(results)
        assert len(trades) == 0
        assert len(order_manager.open_positions) == 0

    def test_multiple_signals(self, order_manager):
        results = [
            _make_risk_result(Direction.LONG, 5000.0),
            _make_risk_result(Direction.SHORT, 5020.0),
        ]
        trades = order_manager.process_signals(results)
        assert len(trades) == 2
        assert len(order_manager.open_positions) == 2

    def test_risk_manager_position_count_updated(self, order_manager, risk_manager):
        results = [_make_risk_result()]
        order_manager.process_signals(results)
        assert risk_manager.open_positions == 1


class TestOnBar:
    def test_stop_loss_exit(self, order_manager):
        # Open a long position at 5000
        order_manager.process_signals([_make_risk_result(Direction.LONG, 5000.0)])
        assert len(order_manager.open_positions) == 1

        # Price drops to stop loss (4996)
        bar = _make_bar(close=4995.0)
        closed = order_manager.on_bar(bar)
        assert len(closed) == 1
        assert closed[0].status == TradeStatus.CLOSED
        assert len(order_manager.open_positions) == 0

    def test_take_profit_exit(self, order_manager):
        # Open a long position at 5000, TP at 5008
        order_manager.process_signals([_make_risk_result(Direction.LONG, 5000.0)])

        # Price rises to take profit
        bar = _make_bar(close=5009.0)
        closed = order_manager.on_bar(bar)
        assert len(closed) == 1
        assert closed[0].status == TradeStatus.CLOSED

    def test_no_exit_when_price_between_stop_and_target(self, order_manager):
        order_manager.process_signals([_make_risk_result(Direction.LONG, 5000.0)])

        bar = _make_bar(close=5003.0)
        closed = order_manager.on_bar(bar)
        assert len(closed) == 0
        assert len(order_manager.open_positions) == 1

    def test_short_stop_loss_exit(self, order_manager):
        order_manager.process_signals([_make_risk_result(Direction.SHORT, 5000.0)])

        # Price rises above stop (5004)
        bar = _make_bar(close=5005.0)
        closed = order_manager.on_bar(bar)
        assert len(closed) == 1

    def test_short_take_profit_exit(self, order_manager):
        order_manager.process_signals([_make_risk_result(Direction.SHORT, 5000.0)])

        # Price drops to take profit (4992)
        bar = _make_bar(close=4991.0)
        closed = order_manager.on_bar(bar)
        assert len(closed) == 1

    def test_risk_manager_updated_on_close(self, order_manager, risk_manager):
        order_manager.process_signals([_make_risk_result(Direction.LONG, 5000.0)])
        assert risk_manager.open_positions == 1

        bar = _make_bar(close=4995.0)  # Trigger stop
        order_manager.on_bar(bar)
        assert risk_manager.open_positions == 0


class TestForceClose:
    def test_force_close_all(self, order_manager):
        order_manager.process_signals([
            _make_risk_result(Direction.LONG, 5000.0),
            _make_risk_result(Direction.SHORT, 5020.0),
        ])
        assert len(order_manager.open_positions) == 2

        closed = order_manager.force_close_all(5005.0)
        assert len(closed) == 2
        assert len(order_manager.open_positions) == 0

    def test_force_close_empty(self, order_manager):
        closed = order_manager.force_close_all(5000.0)
        assert len(closed) == 0


class TestUnrealizedPnl:
    def test_unrealized_pnl_calculation(self, order_manager):
        order_manager.process_signals([_make_risk_result(Direction.LONG, 5000.0)])

        # Update price
        bar = _make_bar(close=5004.0)
        order_manager.on_bar(bar)

        pnl = order_manager.get_total_unrealized_pnl()
        assert pnl > 0

    def test_unrealized_pnl_zero_no_positions(self, order_manager):
        assert order_manager.get_total_unrealized_pnl() == 0.0


class TestWithRecorder:
    def test_closed_trades_recorded(self, executor, risk_manager):
        recorder = TradeRecorder(sqlite_engine=None)
        mgr = OrderManager(
            executor=executor,
            risk_manager=risk_manager,
            recorder=recorder,
        )

        mgr.process_signals([_make_risk_result(Direction.LONG, 5000.0)])
        bar = _make_bar(close=4995.0)
        mgr.on_bar(bar)

        assert len(recorder._today_trades) == 1
