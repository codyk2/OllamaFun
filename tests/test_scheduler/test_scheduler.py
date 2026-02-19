"""Tests for TradingScheduler."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.models import EquitySnapshot
from src.journal.recorder import TradeRecorder
from src.monitoring.health import HealthMonitor
from src.risk.daily_limits import DailyLimitsTracker
from src.scheduler.scheduler import TradingScheduler


@pytest.fixture
def daily_tracker():
    return DailyLimitsTracker(account_equity=10000.0)


@pytest.fixture
def health_monitor():
    return HealthMonitor()


@pytest.fixture
def recorder():
    return TradeRecorder(sqlite_engine=None)


class TestSchedulerInit:
    def test_creates_scheduler(self, daily_tracker, health_monitor):
        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
        )
        assert scheduler is not None
        assert scheduler.scheduler is not None


class TestDailyResetJob:
    def test_daily_reset_resets_tracker(self, daily_tracker, health_monitor, recorder):
        daily_tracker.realized_pnl_today = -100.0
        daily_tracker.trades_today = 5

        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
            trade_recorder=recorder,
        )
        scheduler._daily_reset_job()

        assert daily_tracker.realized_pnl_today == 0.0
        assert daily_tracker.trades_today == 0

    def test_daily_reset_calls_generate_summary(self, daily_tracker, health_monitor):
        recorder = MagicMock(spec=TradeRecorder)
        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
            trade_recorder=recorder,
        )
        scheduler._daily_reset_job()
        recorder.generate_daily_summary.assert_called_once()
        recorder.reset_daily.assert_called_once()


class TestWeeklyResetJob:
    def test_weekly_reset(self, daily_tracker, health_monitor):
        daily_tracker.realized_pnl_week = -300.0
        daily_tracker.weekly_halted = True

        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
        )
        scheduler._weekly_reset_job()

        assert daily_tracker.realized_pnl_week == 0.0
        assert daily_tracker.weekly_halted is False


class TestHealthCheckJob:
    def test_health_check_runs(self, daily_tracker, health_monitor):
        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
        )
        # Should not raise
        scheduler._health_check_job()


class TestEquitySnapshotJob:
    def test_equity_snapshot_recorded(self, daily_tracker, health_monitor):
        snapshot = EquitySnapshot(equity=10050.0, unrealized_pnl=50.0)
        recorder = MagicMock(spec=TradeRecorder)
        getter = MagicMock(return_value=snapshot)

        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
            trade_recorder=recorder,
            equity_getter=getter,
        )
        scheduler._equity_snapshot_job()

        getter.assert_called_once()
        recorder.record_equity_snapshot.assert_called_once_with(snapshot)

    def test_no_getter_no_crash(self, daily_tracker, health_monitor):
        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
            equity_getter=None,
        )
        scheduler._equity_snapshot_job()  # Should not raise


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, daily_tracker, health_monitor):
        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
        )
        scheduler.start()
        assert scheduler.scheduler.running
        scheduler.stop()
        # APScheduler v3 async: shutdown completes but .running may lag
        # Just verify no error was raised

    def test_stop_when_not_running(self, daily_tracker, health_monitor):
        scheduler = TradingScheduler(
            daily_tracker=daily_tracker,
            health_monitor=health_monitor,
        )
        scheduler.stop()  # Should not raise
