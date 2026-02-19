"""Tests for PaperExecutor."""

import random
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.core.models import (
    Direction,
    Position,
    RiskDecision,
    RiskResult,
    Signal,
    Trade,
    TradeStatus,
)
from src.execution.paper_executor import PaperExecutor


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


def _make_position(direction=Direction.LONG, entry_price=5000.0):
    trade = Trade(
        strategy="mean_reversion",
        direction=direction,
        entry_price=entry_price,
        stop_loss=entry_price - 4 if direction == Direction.LONG else entry_price + 4,
        take_profit=entry_price + 8 if direction == Direction.LONG else entry_price - 8,
        quantity=1,
        entry_time=datetime.now(UTC),
        status=TradeStatus.OPEN,
    )
    return Position(trade=trade, current_price=entry_price)


class TestPaperExecutorInit:
    def test_creates_in_paper_mode(self):
        executor = PaperExecutor(paper_mode=True)
        assert executor is not None

    def test_raises_when_not_paper_mode(self):
        with pytest.raises(RuntimeError, match="paper_mode"):
            PaperExecutor(paper_mode=False)


class TestExecuteEntry:
    def test_creates_trade_with_correct_fields(self):
        random.seed(42)
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=0, slippage_ticks_std=0)
        result = _make_risk_result(Direction.LONG, 5000.0)
        trade = executor.execute_entry(result)

        assert trade is not None
        assert trade.direction == Direction.LONG
        assert trade.status == TradeStatus.OPEN
        assert trade.quantity == 1
        assert trade.entry_time is not None

    def test_long_entry_slippage_is_adverse(self):
        """Long entries should fill at or above the signal price."""
        random.seed(42)
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=1.0, slippage_ticks_std=0.0)
        result = _make_risk_result(Direction.LONG, 5000.0)
        trade = executor.execute_entry(result)
        assert trade.entry_price >= 5000.0

    def test_short_entry_slippage_is_adverse(self):
        """Short entries should fill at or below the signal price."""
        random.seed(42)
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=1.0, slippage_ticks_std=0.0)
        result = _make_risk_result(Direction.SHORT, 5000.0)
        trade = executor.execute_entry(result)
        assert trade.entry_price <= 5000.0

    def test_zero_slippage(self):
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=0, slippage_ticks_std=0)
        result = _make_risk_result(Direction.LONG, 5000.0)
        trade = executor.execute_entry(result)
        assert trade.entry_price == 5000.0

    def test_none_signal_returns_none(self):
        executor = PaperExecutor(paper_mode=True)
        result = RiskResult(decision=RiskDecision.APPROVED, position_size=1, signal=None)
        trade = executor.execute_entry(result)
        assert trade is None

    def test_fill_probability(self):
        random.seed(42)
        executor = PaperExecutor(paper_mode=True, fill_probability=0.0)
        result = _make_risk_result()
        trade = executor.execute_entry(result)
        assert trade is None

    def test_commission_applied(self):
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=0, slippage_ticks_std=0)
        result = _make_risk_result(qty=2)
        trade = executor.execute_entry(result)
        assert trade.commission == pytest.approx(1.24 * 2)

    def test_slippage_ticks_recorded(self):
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=2, slippage_ticks_std=0)
        result = _make_risk_result(Direction.LONG, 5000.0)
        trade = executor.execute_entry(result)
        assert trade.slippage_ticks >= 0


class TestExecuteExit:
    def test_exit_closes_trade(self):
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=0, slippage_ticks_std=0)
        position = _make_position(Direction.LONG, 5000.0)
        trade = executor.execute_exit(position, 5008.0, reason="take_profit")
        assert trade.status == TradeStatus.CLOSED
        assert trade.exit_time is not None
        assert trade.notes == "take_profit"

    def test_pnl_calculated_on_exit(self):
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=0, slippage_ticks_std=0)
        position = _make_position(Direction.LONG, 5000.0)
        trade = executor.execute_exit(position, 5008.0)
        assert trade.pnl_ticks is not None
        assert trade.pnl_ticks > 0  # 8 points profit
        assert trade.pnl_dollars is not None

    def test_exit_slippage_adverse_for_long(self):
        """Long exit should fill at or below the target price."""
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=1.0, slippage_ticks_std=0.0)
        position = _make_position(Direction.LONG, 5000.0)
        trade = executor.execute_exit(position, 5008.0)
        assert trade.exit_price <= 5008.0

    def test_exit_slippage_adverse_for_short(self):
        """Short exit should fill at or above the target price."""
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=1.0, slippage_ticks_std=0.0)
        position = _make_position(Direction.SHORT, 5000.0)
        trade = executor.execute_exit(position, 4992.0)
        assert trade.exit_price >= 4992.0

    def test_risk_reward_calculated(self):
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=0, slippage_ticks_std=0)
        position = _make_position(Direction.LONG, 5000.0)
        trade = executor.execute_exit(position, 5008.0)
        assert trade.risk_reward_actual is not None
        assert trade.risk_reward_actual > 0


class TestTickRounding:
    def test_prices_aligned_to_tick_size(self):
        executor = PaperExecutor(paper_mode=True, slippage_ticks_mean=0.5, slippage_ticks_std=0.1)
        random.seed(42)
        result = _make_risk_result(Direction.LONG, 5000.0)
        trade = executor.execute_entry(result)
        # Price should be a multiple of 0.25
        remainder = trade.entry_price % 0.25
        assert abs(remainder) < 0.001 or abs(remainder - 0.25) < 0.001
