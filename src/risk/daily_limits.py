"""Daily and weekly P&L tracking with automatic trading halt.

Tracks realized + unrealized P&L and enforces loss limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import RISK_DEFAULTS
from src.core.models import RiskEvent, Severity


@dataclass
class DailyLimitsTracker:
    """Tracks daily/weekly P&L and enforces loss limits."""

    account_equity: float
    daily_loss_limit_pct: float = RISK_DEFAULTS["daily_loss_limit"]
    weekly_loss_limit_pct: float = RISK_DEFAULTS["weekly_loss_limit"]
    max_daily_trades: int = RISK_DEFAULTS["max_daily_trades"]
    cooldown_seconds: int = RISK_DEFAULTS["cooldown_after_loss"]

    # Internal state
    realized_pnl_today: float = 0.0
    realized_pnl_week: float = 0.0
    unrealized_pnl: float = 0.0
    trades_today: int = 0
    daily_halted: bool = False
    weekly_halted: bool = False
    last_loss_time: float | None = None
    events: list[RiskEvent] = field(default_factory=list)

    @property
    def daily_loss_limit_dollars(self) -> float:
        return self.account_equity * self.daily_loss_limit_pct

    @property
    def weekly_loss_limit_dollars(self) -> float:
        return self.account_equity * self.weekly_loss_limit_pct

    @property
    def total_pnl_today(self) -> float:
        return self.realized_pnl_today + self.unrealized_pnl

    @property
    def is_halted(self) -> bool:
        return self.daily_halted or self.weekly_halted

    @property
    def is_in_cooldown(self) -> bool:
        if self.last_loss_time is None:
            return False
        elapsed = time.monotonic() - self.last_loss_time
        return elapsed < self.cooldown_seconds

    @property
    def cooldown_remaining(self) -> float:
        if self.last_loss_time is None:
            return 0.0
        elapsed = time.monotonic() - self.last_loss_time
        remaining = self.cooldown_seconds - elapsed
        return max(remaining, 0.0)

    def record_trade_closed(self, pnl_dollars: float) -> None:
        """Record a closed trade's P&L and check limits."""
        self.realized_pnl_today += pnl_dollars
        self.realized_pnl_week += pnl_dollars
        self.trades_today += 1

        if pnl_dollars < 0:
            self.last_loss_time = time.monotonic()

        self._check_daily_limit()
        self._check_weekly_limit()

    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update unrealized P&L and check limits."""
        self.unrealized_pnl = unrealized_pnl
        self._check_daily_limit()
        self._check_weekly_limit()

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed. Returns (allowed, reason)."""
        if self.daily_halted:
            return False, "Daily loss limit reached"
        if self.weekly_halted:
            return False, "Weekly loss limit reached"
        if self.trades_today >= self.max_daily_trades:
            return False, f"Max daily trades reached ({self.max_daily_trades})"
        if self.is_in_cooldown:
            return False, f"Cooldown active ({self.cooldown_remaining:.0f}s remaining)"
        return True, ""

    def reset_daily(self) -> None:
        """Reset daily tracking at session boundary."""
        self.realized_pnl_today = 0.0
        self.unrealized_pnl = 0.0
        self.trades_today = 0
        self.daily_halted = False
        self.last_loss_time = None
        self.events.clear()

    def reset_weekly(self) -> None:
        """Reset weekly tracking."""
        self.realized_pnl_week = 0.0
        self.weekly_halted = False
        self.reset_daily()

    def _check_daily_limit(self) -> None:
        """Check if daily loss limit has been breached."""
        if self.daily_halted:
            return
        if self.total_pnl_today <= -self.daily_loss_limit_dollars:
            self.daily_halted = True
            self.events.append(RiskEvent(
                event_type="DAILY_LIMIT",
                details={
                    "realized_pnl": self.realized_pnl_today,
                    "unrealized_pnl": self.unrealized_pnl,
                    "total_pnl": self.total_pnl_today,
                    "limit": -self.daily_loss_limit_dollars,
                },
                severity=Severity.CRITICAL,
            ))

    def _check_weekly_limit(self) -> None:
        """Check if weekly loss limit has been breached."""
        if self.weekly_halted:
            return
        total_week = self.realized_pnl_week + self.unrealized_pnl
        if total_week <= -self.weekly_loss_limit_dollars:
            self.weekly_halted = True
            self.events.append(RiskEvent(
                event_type="WEEKLY_LIMIT",
                details={
                    "realized_pnl_week": self.realized_pnl_week,
                    "unrealized_pnl": self.unrealized_pnl,
                    "total_pnl_week": total_week,
                    "limit": -self.weekly_loss_limit_dollars,
                },
                severity=Severity.CRITICAL,
            ))
