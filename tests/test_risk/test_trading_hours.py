"""Tests for the active trading window (10AM-2PM ET restriction)."""

from datetime import datetime, time

import pytz
import pytest

from src.core.models import Direction, RiskDecision, Signal
from src.risk.manager import RiskManager
from src.strategies.base import TradingWindow

ET = pytz.timezone("America/New_York")
CT = pytz.timezone("America/Chicago")


def _make_signal(
    direction=Direction.LONG,
    entry=5000.00,
    stop=4996.00,
    target=5004.75,
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
def window():
    return TradingWindow(start_et=time(10, 0), end_et=time(14, 0))


class TestTradingWindowApproval:
    def test_signal_allowed_at_10_30_am_et(self, manager, window):
        """10:30 AM ET is within the 10AM-2PM window."""
        t = ET.localize(datetime(2025, 1, 15, 10, 30))  # Wednesday
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.APPROVED

    def test_signal_allowed_at_noon_et(self, manager, window):
        """12:00 PM ET is within the window."""
        t = ET.localize(datetime(2025, 1, 15, 12, 0))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.APPROVED

    def test_signal_allowed_at_1_59_pm_et(self, manager, window):
        """1:59 PM ET is still within the window."""
        t = ET.localize(datetime(2025, 1, 15, 13, 59))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.APPROVED

    def test_signal_allowed_exactly_at_start(self, manager, window):
        """10:00 AM ET is the start of the window (inclusive)."""
        t = ET.localize(datetime(2025, 1, 15, 10, 0))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.APPROVED


class TestTradingWindowRejection:
    def test_signal_blocked_at_9_35_am_et(self, manager, window):
        """9:35 AM ET is before the 10AM start."""
        t = ET.localize(datetime(2025, 1, 15, 9, 35))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.REJECTED
        assert "trading window" in result.reason.lower()

    def test_signal_blocked_at_2_15_pm_et(self, manager, window):
        """2:15 PM ET is after the 2PM end."""
        t = ET.localize(datetime(2025, 1, 15, 14, 15))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.REJECTED
        assert "trading window" in result.reason.lower()

    def test_signal_blocked_exactly_at_end(self, manager, window):
        """2:00 PM ET is the end of the window (exclusive)."""
        t = ET.localize(datetime(2025, 1, 15, 14, 0))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.REJECTED
        assert "trading window" in result.reason.lower()

    def test_signal_blocked_overnight(self, manager, window):
        """8:00 PM ET overnight session is outside the window."""
        t = ET.localize(datetime(2025, 1, 15, 20, 0))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.REJECTED
        assert "trading window" in result.reason.lower()


class TestTradingWindowTimezoneConversion:
    def test_ct_time_converted_to_et(self, manager, window):
        """CT time should be converted to ET for window check (CT = ET - 1hr)."""
        # 9:30 AM CT = 10:30 AM ET -> within window
        t = CT.localize(datetime(2025, 1, 15, 9, 30))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.APPROVED

    def test_ct_before_window_rejected(self, manager, window):
        """8:30 AM CT = 9:30 AM ET -> before window."""
        t = CT.localize(datetime(2025, 1, 15, 8, 30))
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=window)
        assert result.decision == RiskDecision.REJECTED


class TestNoTradingWindow:
    def test_no_window_allows_any_time(self, manager):
        """Without a trading window, only CME hours apply."""
        t = ET.localize(datetime(2025, 1, 15, 20, 0))  # 8 PM ET
        signal = _make_signal()
        result = manager.evaluate(signal, atr=3.0, current_time=t, trading_window=None)
        assert result.decision == RiskDecision.APPROVED
