"""Risk manager — the central gatekeeper for all trading decisions.

Every signal must pass through this module before execution.
It checks: position sizing, stop validity, R:R ratio, daily limits,
concurrent positions, trading hours, and cooldown timer.
"""

from __future__ import annotations

from datetime import datetime, time

import pytz

from src.config import RISK_DEFAULTS, MES_SPEC
from src.core.models import (
    Direction,
    RiskDecision,
    RiskEvent,
    RiskResult,
    Severity,
    Signal,
)
from src.risk.daily_limits import DailyLimitsTracker
from src.risk.position_sizer import calculate_position_size, validate_stop_distance
from src.risk.stop_loss import (
    calculate_risk_reward_ratio,
    calculate_stop_distance_ticks,
    validate_stop_placement,
)

CT = pytz.timezone("America/Chicago")


class RiskManager:
    """Orchestrates all risk checks for incoming signals."""

    def __init__(
        self,
        account_equity: float,
        daily_tracker: DailyLimitsTracker | None = None,
        risk_config: dict | None = None,
        contract_spec: dict | None = None,
    ) -> None:
        self.account_equity = account_equity
        self.config = {**RISK_DEFAULTS, **(risk_config or {})}
        self.spec = contract_spec or MES_SPEC
        self.daily_tracker = daily_tracker or DailyLimitsTracker(
            account_equity=account_equity,
            daily_loss_limit_pct=self.config["daily_loss_limit"],
            weekly_loss_limit_pct=self.config["weekly_loss_limit"],
            max_daily_trades=self.config["max_daily_trades"],
            cooldown_seconds=self.config["cooldown_after_loss"],
        )
        self.open_positions: int = 0
        self.events: list[RiskEvent] = []

    def evaluate(
        self,
        signal: Signal,
        atr: float,
        current_time: datetime | None = None,
    ) -> RiskResult:
        """Evaluate a trading signal against all risk rules.

        Returns RiskResult with APPROVED (and position size) or REJECTED (with reason).
        """
        # 1. Check if trading is halted (daily/weekly limits)
        can_trade, reason = self.daily_tracker.can_trade()
        if not can_trade:
            return self._reject(signal, reason, "DAILY_LIMIT_CHECK")

        # 2. Check trading hours
        now = current_time or datetime.now(CT)
        hours_ok, hours_reason = self._check_trading_hours(now)
        if not hours_ok:
            return self._reject(signal, hours_reason, "TRADING_HOURS_CHECK")

        # 3. Check concurrent positions
        if self.open_positions >= self.config["max_concurrent_positions"]:
            return self._reject(
                signal,
                f"Max concurrent positions reached ({self.config['max_concurrent_positions']})",
                "MAX_POSITIONS_CHECK",
            )

        # 4. Validate stop-loss exists and placement
        if self.config["always_use_stop_loss"] and signal.stop_loss is None:
            return self._reject(signal, "Stop-loss is required", "STOP_REQUIRED_CHECK")

        stop_valid, stop_reason = validate_stop_placement(
            entry_price=signal.entry_price,
            stop_price=signal.stop_loss,
            direction=signal.direction,
            atr=atr,
            max_atr_multiple=self.config["max_stop_distance_atr"],
            tick_size=self.spec["tick_size"],
        )
        if not stop_valid:
            return self._reject(signal, stop_reason, "STOP_VALIDATION_CHECK")

        # 5. Check stop distance vs ATR limit
        stop_ticks = calculate_stop_distance_ticks(
            entry_price=signal.entry_price,
            stop_price=signal.stop_loss,
            direction=signal.direction,
            tick_size=self.spec["tick_size"],
        )
        if not validate_stop_distance(
            stop_distance_ticks=stop_ticks,
            atr_ticks=atr / self.spec["tick_size"],
            max_atr_multiple=self.config["max_stop_distance_atr"],
        ):
            return self._reject(
                signal,
                f"Stop distance ({stop_ticks} ticks) exceeds ATR limit",
                "ATR_DISTANCE_CHECK",
            )

        # 6. Check risk:reward ratio
        if signal.take_profit is not None:
            rr_ratio = calculate_risk_reward_ratio(
                entry_price=signal.entry_price,
                stop_price=signal.stop_loss,
                target_price=signal.take_profit,
                direction=signal.direction,
            )
            if rr_ratio < self.config["min_risk_reward_ratio"]:
                return self._reject(
                    signal,
                    f"R:R ratio ({rr_ratio:.2f}) below minimum ({self.config['min_risk_reward_ratio']})",
                    "RR_RATIO_CHECK",
                )

        # 7. Calculate position size
        position_size = calculate_position_size(
            account_equity=self.account_equity,
            stop_distance_ticks=stop_ticks,
            tick_value=self.spec["tick_value"],
            max_risk_pct=self.config["max_risk_per_trade"],
            max_position_size=self.config["max_position_size"],
        )
        if position_size <= 0:
            return self._reject(
                signal,
                "Position size calculated as 0 (insufficient equity or stop too wide)",
                "POSITION_SIZE_CHECK",
            )

        # All checks passed
        return RiskResult(
            decision=RiskDecision.APPROVED,
            position_size=position_size,
            reason="All risk checks passed",
            signal=signal,
        )

    def record_position_opened(self) -> None:
        """Track that a new position has been opened."""
        self.open_positions += 1

    def record_position_closed(self, pnl_dollars: float) -> None:
        """Track that a position has been closed."""
        self.open_positions = max(0, self.open_positions - 1)
        self.daily_tracker.record_trade_closed(pnl_dollars)

    def update_equity(self, new_equity: float) -> None:
        """Update account equity."""
        self.account_equity = new_equity
        self.daily_tracker.account_equity = new_equity

    def _check_trading_hours(self, now: datetime) -> tuple[bool, str]:
        """Check if current time is within trading hours."""
        if now.tzinfo is None:
            now = CT.localize(now)
        else:
            now = now.astimezone(CT)

        current_time = now.time()
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # CME Globex: Sunday 5PM to Friday 4PM CT
        # Saturday = completely closed
        if weekday == 5:  # Saturday
            return False, "Market closed (Saturday)"

        # Sunday before 5PM
        if weekday == 6 and current_time < time(17, 0):
            return False, "Market closed (Sunday before 5PM CT)"

        # Friday after 4PM
        if weekday == 4 and current_time >= time(16, 0):
            return False, "Market closed (Friday after 4PM CT)"

        # Daily maintenance break: 4:00 PM - 5:00 PM CT (Mon-Thu)
        if time(16, 0) <= current_time < time(17, 0) and weekday in (0, 1, 2, 3):
            return False, "Daily maintenance break (4-5PM CT)"

        # Skip first N minutes of session
        skip_first = self.config["skip_first_minutes"]
        session_start = time(17, skip_first)
        if current_time >= time(17, 0) and current_time < session_start:
            return False, f"Skipping first {skip_first} minutes of session"

        # Skip last N minutes before close
        skip_last = self.config["skip_last_minutes"]
        close_buffer = time(15, 60 - skip_last)
        if weekday == 4 and current_time >= close_buffer:
            return False, f"Skipping last {skip_last} minutes before close"

        return True, ""

    def _reject(self, signal: Signal, reason: str, check_name: str) -> RiskResult:
        """Create a rejection result and log the event."""
        self.events.append(RiskEvent(
            event_type=f"SIGNAL_REJECTED:{check_name}",
            details={
                "strategy": signal.strategy,
                "direction": signal.direction.value,
                "entry": signal.entry_price,
                "stop": signal.stop_loss,
                "reason": reason,
            },
            severity=Severity.WARNING,
        ))
        return RiskResult(
            decision=RiskDecision.REJECTED,
            position_size=0,
            reason=reason,
            signal=signal,
        )
