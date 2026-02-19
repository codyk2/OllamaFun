"""Scheduled jobs using APScheduler.

Jobs:
  - daily_reset: Reset daily P&L tracking at 5PM CT (session boundary)
  - weekly_reset: Reset weekly tracking on Sunday 5PM CT
  - health_check: Run health checks every 60 seconds
  - equity_snapshot: Record equity every 5 minutes during market hours
"""

from __future__ import annotations

from typing import Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.core.logging import get_logger
from src.core.models import EquitySnapshot
from src.journal.recorder import TradeRecorder
from src.monitoring.health import HealthMonitor
from src.risk.daily_limits import DailyLimitsTracker

logger = get_logger("scheduler")


class TradingScheduler:
    """Manages scheduled jobs for the trading system."""

    def __init__(
        self,
        daily_tracker: DailyLimitsTracker,
        health_monitor: HealthMonitor,
        trade_recorder: TradeRecorder | None = None,
        equity_getter: Callable[[], EquitySnapshot] | None = None,
    ) -> None:
        self.daily_tracker = daily_tracker
        self.health = health_monitor
        self.recorder = trade_recorder
        self.equity_getter = equity_getter
        self.scheduler = AsyncIOScheduler(timezone="America/Chicago")

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        # Daily reset at 5PM CT (Mon-Fri)
        self.scheduler.add_job(
            self._daily_reset_job,
            "cron",
            hour=17,
            minute=0,
            day_of_week="mon-fri",
            id="daily_reset",
        )

        # Weekly reset on Sunday at 5PM CT
        self.scheduler.add_job(
            self._weekly_reset_job,
            "cron",
            hour=17,
            minute=0,
            day_of_week="sun",
            id="weekly_reset",
        )

        # Health check every 60 seconds
        self.scheduler.add_job(
            self._health_check_job,
            "interval",
            seconds=60,
            id="health_check",
        )

        # Equity snapshot every 5 minutes
        if self.equity_getter:
            self.scheduler.add_job(
                self._equity_snapshot_job,
                "interval",
                minutes=5,
                id="equity_snapshot",
            )

        self.scheduler.start()
        logger.info("scheduler_started")

    def stop(self) -> None:
        """Shut down the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("scheduler_stopped")

    def _daily_reset_job(self) -> None:
        """Reset daily P&L tracking and generate daily summary."""
        logger.info("daily_reset_running")

        # Generate daily summary before reset
        if self.recorder:
            self.recorder.generate_daily_summary()

        self.daily_tracker.reset_daily()

        if self.recorder:
            self.recorder.reset_daily()

        logger.info("daily_reset_complete")

    def _weekly_reset_job(self) -> None:
        """Reset weekly P&L tracking."""
        logger.info("weekly_reset_running")
        self.daily_tracker.reset_weekly()
        logger.info("weekly_reset_complete")

    def _health_check_job(self) -> None:
        """Run periodic health check."""
        status = self.health.check_all()
        if status.overall_status.value != "HEALTHY":
            logger.warning(
                "health_check_degraded",
                status=status.overall_status.value,
            )

    def _equity_snapshot_job(self) -> None:
        """Take equity snapshot and persist."""
        if not self.equity_getter or not self.recorder:
            return

        try:
            snapshot = self.equity_getter()
            self.recorder.record_equity_snapshot(snapshot)
        except Exception as e:
            logger.error("equity_snapshot_failed", error=str(e))
