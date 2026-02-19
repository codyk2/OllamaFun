"""Tests for economic event news calendar and blackout windows."""

from datetime import datetime, time

import pytz
import pytest

from src.core.models import Direction, RiskDecision, Signal
from src.risk.manager import RiskManager
from src.signals.news_calendar import (
    CPI_BUFFER,
    EconomicEvent,
    FOMC_BUFFER,
    NFP_BUFFER,
    NewsCalendar,
)

ET = pytz.timezone("America/New_York")
CT = pytz.timezone("America/Chicago")


def _make_signal() -> Signal:
    return Signal(
        strategy="mean_reversion",
        symbol="MES",
        direction=Direction.LONG,
        confidence=0.7,
        entry_price=5000.00,
        stop_loss=4996.00,
        take_profit=5004.75,
        reason="test",
    )


class TestEconomicEvent:
    def test_fomc_blackout_window(self):
        """FOMC: 30 min pre, 60 min post."""
        event_time = ET.localize(datetime(2025, 1, 29, 14, 0))  # 2:00 PM
        event = EconomicEvent(name="FOMC", event_time=event_time, **FOMC_BUFFER)

        # 30 min before = 1:30 PM -> blocked
        assert event.is_blocked(ET.localize(datetime(2025, 1, 29, 13, 30))) is True
        # 5 min before = 1:55 PM -> blocked
        assert event.is_blocked(ET.localize(datetime(2025, 1, 29, 13, 55))) is True
        # 30 min after = 2:30 PM -> blocked
        assert event.is_blocked(ET.localize(datetime(2025, 1, 29, 14, 30))) is True
        # 60 min after = 3:00 PM -> blocked (edge)
        assert event.is_blocked(ET.localize(datetime(2025, 1, 29, 15, 0))) is True
        # 61 min after = 3:01 PM -> not blocked
        assert event.is_blocked(ET.localize(datetime(2025, 1, 29, 15, 1))) is False
        # 31 min before = 1:29 PM -> not blocked
        assert event.is_blocked(ET.localize(datetime(2025, 1, 29, 13, 29))) is False

    def test_nfp_blackout_window(self):
        """NFP: 15 min pre, 45 min post."""
        event_time = ET.localize(datetime(2025, 2, 7, 8, 30))  # 8:30 AM
        event = EconomicEvent(name="NFP", event_time=event_time, **NFP_BUFFER)

        # 15 min before = 8:15 AM -> blocked
        assert event.is_blocked(ET.localize(datetime(2025, 2, 7, 8, 15))) is True
        # 45 min after = 9:15 AM -> blocked
        assert event.is_blocked(ET.localize(datetime(2025, 2, 7, 9, 15))) is True
        # 46 min after = 9:16 AM -> not blocked
        assert event.is_blocked(ET.localize(datetime(2025, 2, 7, 9, 16))) is False

    def test_cpi_blackout_window(self):
        """CPI: 15 min pre, 30 min post."""
        event_time = ET.localize(datetime(2025, 1, 15, 8, 30))
        event = EconomicEvent(name="CPI", event_time=event_time, **CPI_BUFFER)

        assert event.is_blocked(ET.localize(datetime(2025, 1, 15, 8, 15))) is True
        assert event.is_blocked(ET.localize(datetime(2025, 1, 15, 9, 0))) is True
        assert event.is_blocked(ET.localize(datetime(2025, 1, 15, 9, 1))) is False


class TestNewsCalendar:
    def test_empty_calendar_not_blocked(self):
        calendar = NewsCalendar()
        blocked, reason = calendar.is_blocked(ET.localize(datetime(2025, 1, 15, 10, 0)))
        assert blocked is False
        assert reason == ""

    def test_blocked_during_fomc(self):
        calendar = NewsCalendar()
        calendar.add_fomc(ET.localize(datetime(2025, 1, 29, 14, 0)))

        blocked, reason = calendar.is_blocked(ET.localize(datetime(2025, 1, 29, 14, 30)))
        assert blocked is True
        assert "FOMC" in reason

    def test_not_blocked_away_from_events(self):
        calendar = NewsCalendar()
        calendar.add_fomc(ET.localize(datetime(2025, 1, 29, 14, 0)))

        blocked, _ = calendar.is_blocked(ET.localize(datetime(2025, 1, 29, 10, 0)))
        assert blocked is False

    def test_next_event(self):
        calendar = NewsCalendar()
        fomc_time = ET.localize(datetime(2025, 1, 29, 14, 0))
        nfp_time = ET.localize(datetime(2025, 2, 7, 8, 30))
        calendar.add_fomc(fomc_time)
        calendar.add_nfp(nfp_time)

        now = ET.localize(datetime(2025, 1, 28, 10, 0))
        next_event = calendar.next_event(now)
        assert next_event is not None
        assert next_event.name == "FOMC"

    def test_next_event_after_first(self):
        calendar = NewsCalendar()
        fomc_time = ET.localize(datetime(2025, 1, 29, 14, 0))
        nfp_time = ET.localize(datetime(2025, 2, 7, 8, 30))
        calendar.add_fomc(fomc_time)
        calendar.add_nfp(nfp_time)

        now = ET.localize(datetime(2025, 1, 30, 10, 0))
        next_event = calendar.next_event(now)
        assert next_event is not None
        assert next_event.name == "NFP"

    def test_no_next_event_when_all_past(self):
        calendar = NewsCalendar()
        calendar.add_fomc(ET.localize(datetime(2025, 1, 29, 14, 0)))

        now = ET.localize(datetime(2025, 2, 1, 10, 0))
        assert calendar.next_event(now) is None

    def test_clear_past_events(self):
        calendar = NewsCalendar()
        calendar.add_fomc(ET.localize(datetime(2025, 1, 29, 14, 0)))
        calendar.add_nfp(ET.localize(datetime(2025, 2, 7, 8, 30)))

        now = ET.localize(datetime(2025, 2, 1, 10, 0))
        removed = calendar.clear_past_events(now)
        assert removed == 1
        assert len(calendar.events) == 1


class TestNewsCalendarIntegration:
    def test_risk_manager_blocks_during_fomc(self):
        """Risk manager should reject signals during news blackout."""
        calendar = NewsCalendar()
        calendar.add_fomc(ET.localize(datetime(2025, 1, 15, 14, 0)))

        manager = RiskManager(account_equity=10000, news_calendar=calendar)
        signal = _make_signal()

        # During FOMC blackout (2:30 PM ET = 1:30 PM CT)
        t = ET.localize(datetime(2025, 1, 15, 14, 30))
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.REJECTED
        assert "blackout" in result.reason.lower() or "FOMC" in result.reason

    def test_risk_manager_allows_outside_blackout(self):
        """Risk manager should allow signals outside news blackout."""
        calendar = NewsCalendar()
        calendar.add_fomc(ET.localize(datetime(2025, 1, 15, 14, 0)))

        manager = RiskManager(account_equity=10000, news_calendar=calendar)
        signal = _make_signal()

        # Well outside blackout (10:30 AM ET = 9:30 AM CT)
        t = ET.localize(datetime(2025, 1, 15, 10, 30))
        result = manager.evaluate(signal, atr=3.0, current_time=t)
        assert result.decision == RiskDecision.APPROVED
