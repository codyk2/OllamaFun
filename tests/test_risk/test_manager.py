"""Tests for the risk manager orchestrator."""

from datetime import datetime

import pytz
import pytest

from src.core.models import Direction, RiskDecision, Signal
from src.risk.daily_limits import DailyLimitsTracker
from src.risk.manager import RiskManager

CT = pytz.timezone("America/Chicago")


def _make_signal(
    direction=Direction.LONG,
    entry=5000.00,
    stop=4996.00,
    target=5008.00,
    confidence=0.7,
) -> Signal:
    return Signal(
        strategy="mean_reversion",
        symbol="MES",
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        reason="test signal",
    )


@pytest.fixture
def manager():
    return RiskManager(account_equity=10000)


@pytest.fixture
def trading_time():
    """A valid trading time: Wednesday 10AM CT."""
    return CT.localize(datetime(2025, 1, 15, 10, 0))  # Wednesday


class TestRiskManagerApproval:
    def test_valid_long_signal_approved(self, manager, trading_time):
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.APPROVED
        assert result.position_size >= 1

    def test_valid_short_signal_approved(self, manager, trading_time):
        signal = _make_signal(
            direction=Direction.SHORT,
            entry=5000,
            stop=5004,
            target=4992,
        )
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.APPROVED

    def test_approved_has_positive_position_size(self, manager, trading_time):
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.position_size > 0


class TestRiskManagerRejections:
    def test_reject_when_daily_halted(self, manager, trading_time):
        manager.daily_tracker.daily_halted = True
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "Daily loss limit" in result.reason

    def test_reject_when_weekly_halted(self, manager, trading_time):
        manager.daily_tracker.weekly_halted = True
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "Weekly loss limit" in result.reason

    def test_reject_max_daily_trades(self, manager, trading_time):
        manager.daily_tracker.trades_today = 10
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "Max daily trades" in result.reason

    def test_reject_max_concurrent_positions(self, manager, trading_time):
        manager.open_positions = 2  # max default is 2
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "concurrent positions" in result.reason

    def test_reject_long_stop_above_entry(self, manager, trading_time):
        signal = _make_signal(stop=5002)  # Stop above entry for long
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "below entry" in result.reason

    def test_reject_short_stop_below_entry(self, manager, trading_time):
        signal = _make_signal(
            direction=Direction.SHORT,
            entry=5000,
            stop=4998,
            target=4992,
        )
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "above entry" in result.reason

    def test_reject_stop_too_wide(self, manager, trading_time):
        # ATR = 3.0, max multiple = 2.0, max stop = 6 points
        # Signal has 10-point stop (40 ticks)
        signal = _make_signal(stop=4990, target=5020)
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "exceeds" in result.reason.lower() or "ATR" in result.reason

    def test_reject_bad_risk_reward(self, manager, trading_time):
        # 16 tick stop, 8 tick target = 0.5 R:R (below minimum 0.8)
        signal = _make_signal(target=5002)
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "R:R ratio" in result.reason

    def test_reject_zero_position_size(self, trading_time):
        """Very small account can't afford even 1 contract."""
        manager = RiskManager(account_equity=50)
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert result.decision == RiskDecision.REJECTED
        assert "Position size" in result.reason


class TestTradingHours:
    def test_allow_weekday_morning(self, manager):
        t = CT.localize(datetime(2025, 1, 15, 10, 0))  # Wed 10 AM
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.APPROVED

    def test_allow_weekday_evening(self, manager):
        t = CT.localize(datetime(2025, 1, 15, 20, 0))  # Wed 8 PM
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.APPROVED

    def test_reject_saturday(self, manager):
        t = CT.localize(datetime(2025, 1, 18, 12, 0))  # Saturday
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.REJECTED
        assert "Saturday" in result.reason

    def test_reject_sunday_before_open(self, manager):
        t = CT.localize(datetime(2025, 1, 19, 14, 0))  # Sunday 2 PM
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.REJECTED
        assert "Sunday" in result.reason

    def test_allow_sunday_after_open(self, manager):
        t = CT.localize(datetime(2025, 1, 19, 17, 30))  # Sunday 5:30 PM
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.APPROVED

    def test_reject_friday_after_close(self, manager):
        t = CT.localize(datetime(2025, 1, 17, 16, 30))  # Friday 4:30 PM
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.REJECTED
        assert "Friday" in result.reason

    def test_reject_daily_maintenance(self, manager):
        t = CT.localize(datetime(2025, 1, 15, 16, 30))  # Wed 4:30 PM
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.REJECTED
        assert "maintenance" in result.reason

    def test_reject_first_minutes_of_session(self, manager):
        t = CT.localize(datetime(2025, 1, 15, 17, 3))  # Wed 5:03 PM (within skip)
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.REJECTED
        assert "first" in result.reason.lower()


class TestPositionTracking:
    def test_record_position_opened(self, manager):
        manager.record_position_opened()
        assert manager.open_positions == 1

    def test_record_position_closed(self, manager):
        manager.open_positions = 2
        manager.record_position_closed(pnl_dollars=50.0)
        assert manager.open_positions == 1

    def test_close_doesnt_go_negative(self, manager):
        manager.record_position_closed(pnl_dollars=0)
        assert manager.open_positions == 0

    def test_close_updates_daily_tracker(self, manager):
        manager.record_position_closed(pnl_dollars=-100.0)
        assert manager.daily_tracker.realized_pnl_today == -100.0


class TestEquityUpdate:
    def test_update_equity(self, manager):
        manager.update_equity(15000)
        assert manager.account_equity == 15000
        assert manager.daily_tracker.account_equity == 15000


class TestEventLogging:
    def test_rejection_creates_event(self, manager, trading_time):
        signal = _make_signal(target=5002)  # Bad R:R
        manager.evaluate(signal, atr=3.0, current_time=trading_time)
        assert len(manager.events) >= 1
        assert "SIGNAL_REJECTED" in manager.events[0].event_type
