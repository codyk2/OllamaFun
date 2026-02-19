"""Tests for daily/weekly P&L tracking and trading halt logic."""

import time

import pytest

from src.risk.daily_limits import DailyLimitsTracker


class TestDailyLimitsTracker:
    @pytest.fixture
    def tracker(self):
        return DailyLimitsTracker(
            account_equity=10000,
            daily_loss_limit_pct=0.03,
            weekly_loss_limit_pct=0.06,
            max_daily_trades=10,
            cooldown_seconds=2,  # Short for testing
        )

    def test_initial_state(self, tracker):
        assert tracker.realized_pnl_today == 0.0
        assert tracker.realized_pnl_week == 0.0
        assert tracker.unrealized_pnl == 0.0
        assert tracker.trades_today == 0
        assert not tracker.daily_halted
        assert not tracker.weekly_halted
        assert not tracker.is_halted
        can, reason = tracker.can_trade()
        assert can is True
        assert reason == ""

    def test_daily_loss_limit_dollars(self, tracker):
        assert tracker.daily_loss_limit_dollars == 300.0  # 10000 * 0.03

    def test_weekly_loss_limit_dollars(self, tracker):
        assert tracker.weekly_loss_limit_dollars == 600.0  # 10000 * 0.06

    def test_record_winning_trade(self, tracker):
        tracker.record_trade_closed(50.0)
        assert tracker.realized_pnl_today == 50.0
        assert tracker.realized_pnl_week == 50.0
        assert tracker.trades_today == 1
        assert not tracker.is_halted

    def test_record_losing_trade(self, tracker):
        tracker.record_trade_closed(-50.0)
        assert tracker.realized_pnl_today == -50.0
        assert tracker.trades_today == 1
        assert not tracker.is_halted

    def test_daily_limit_triggers(self, tracker):
        """Breaching daily loss limit halts trading."""
        tracker.record_trade_closed(-200.0)
        assert not tracker.daily_halted

        tracker.record_trade_closed(-100.0)  # Total: -$300 = exactly limit
        assert tracker.daily_halted
        assert tracker.is_halted

        can, reason = tracker.can_trade()
        assert can is False
        assert "Daily loss limit" in reason

    def test_daily_limit_with_unrealized(self, tracker):
        """Unrealized P&L counts toward daily limit."""
        tracker.record_trade_closed(-100.0)
        assert not tracker.daily_halted

        tracker.update_unrealized(-200.0)  # Total: -100 + -200 = -$300
        assert tracker.daily_halted

    def test_weekly_limit_triggers(self, tracker):
        """Breaching weekly loss limit halts trading."""
        for _ in range(6):
            tracker.record_trade_closed(-100.0)
        # Total week: -$600 = exactly weekly limit
        assert tracker.weekly_halted
        assert tracker.is_halted

    def test_weekly_limit_persists_after_daily_reset(self, tracker):
        """Weekly halt persists even after daily reset."""
        for _ in range(6):
            tracker.record_trade_closed(-100.0)
        assert tracker.weekly_halted

        tracker.reset_daily()
        # Daily should be reset but weekly halt remains
        assert not tracker.daily_halted
        # Need to check: weekly_halted still True since we didn't reset weekly
        # Actually reset_daily doesn't reset weekly_halted, only reset_weekly does
        assert tracker.weekly_halted

    def test_max_daily_trades(self, tracker):
        """Cannot exceed max daily trades."""
        for i in range(10):
            tracker.record_trade_closed(10.0)

        can, reason = tracker.can_trade()
        assert can is False
        assert "Max daily trades" in reason

    def test_cooldown_after_loss(self, tracker):
        """Cooldown timer activates after a losing trade."""
        tracker.record_trade_closed(-10.0)

        assert tracker.is_in_cooldown
        can, reason = tracker.can_trade()
        assert can is False
        assert "Cooldown" in reason

    def test_cooldown_expires(self, tracker):
        """Cooldown expires after the configured duration."""
        tracker.record_trade_closed(-10.0)
        assert tracker.is_in_cooldown

        time.sleep(2.1)  # Wait for 2-second cooldown to expire
        assert not tracker.is_in_cooldown
        can, _ = tracker.can_trade()
        assert can is True

    def test_no_cooldown_after_win(self, tracker):
        """No cooldown after a winning trade."""
        tracker.record_trade_closed(50.0)
        assert not tracker.is_in_cooldown

    def test_cooldown_remaining(self, tracker):
        """Cooldown remaining decreases over time."""
        tracker.record_trade_closed(-10.0)
        remaining = tracker.cooldown_remaining
        assert remaining > 0
        assert remaining <= 2.0

    def test_cooldown_remaining_when_not_in_cooldown(self, tracker):
        assert tracker.cooldown_remaining == 0.0

    def test_reset_daily(self, tracker):
        """Daily reset clears all daily state."""
        tracker.record_trade_closed(-100.0)
        tracker.update_unrealized(-50.0)
        tracker.daily_halted = True

        tracker.reset_daily()

        assert tracker.realized_pnl_today == 0.0
        assert tracker.unrealized_pnl == 0.0
        assert tracker.trades_today == 0
        assert not tracker.daily_halted
        assert tracker.last_loss_time is None

    def test_reset_weekly(self, tracker):
        """Weekly reset clears everything."""
        tracker.record_trade_closed(-500.0)
        tracker.weekly_halted = True

        tracker.reset_weekly()

        assert tracker.realized_pnl_week == 0.0
        assert not tracker.weekly_halted
        assert tracker.realized_pnl_today == 0.0  # daily also resets

    def test_total_pnl_today(self, tracker):
        """Total P&L includes both realized and unrealized."""
        tracker.record_trade_closed(100.0)
        tracker.update_unrealized(-30.0)
        assert tracker.total_pnl_today == 70.0

    def test_events_logged_on_daily_halt(self, tracker):
        """Risk events are created when limits trigger."""
        tracker.record_trade_closed(-300.0)
        assert len(tracker.events) == 1
        assert tracker.events[0].event_type == "DAILY_LIMIT"
        assert tracker.events[0].severity.value == "CRITICAL"

    def test_events_logged_on_weekly_halt(self, tracker):
        for _ in range(6):
            tracker.record_trade_closed(-100.0)
        weekly_events = [e for e in tracker.events if e.event_type == "WEEKLY_LIMIT"]
        assert len(weekly_events) == 1

    def test_double_halt_doesnt_duplicate_events(self, tracker):
        """Hitting the limit twice doesn't create duplicate events."""
        tracker.record_trade_closed(-300.0)
        assert tracker.daily_halted
        events_count = len(tracker.events)

        tracker.record_trade_closed(-50.0)  # Further loss
        assert len(tracker.events) == events_count  # No new event

    def test_multiple_small_losses_accumulate(self, tracker):
        """Many small losses can trigger the daily limit."""
        for _ in range(30):
            tracker.record_trade_closed(-10.0)
        assert tracker.daily_halted
        assert tracker.realized_pnl_today == -300.0
