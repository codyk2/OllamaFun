"""Apex Trader Funding trailing drawdown tracker.

Tracks the trailing drawdown rule used by Apex prop firm accounts.
The drawdown floor ratchets upward with equity but never moves down.
If equity touches or drops below the floor, the account is busted.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ApexDrawdownTracker:
    """Tracks Apex Trader Funding trailing drawdown and profit goal.

    Apex rule: Trailing drawdown follows max equity upward, never moves down.
    If equity drops below (max_equity - trailing_threshold), account is busted.

    Example: $50K account, $2.5K trailing drawdown, $3K profit goal.
    - Start: floor = $47,500
    - Make $1K: max equity = $51K, floor = $48.5K
    - Floor only moves UP with equity, never down.
    - Eval passes when realized P&L >= profit_goal ($3K).
    """

    account_id: str
    trailing_threshold: float  # e.g., 2500.0 for $50K Apex
    max_equity_high: float  # High water mark
    current_equity: float
    starting_equity: float = 50000.0
    profit_goal: float = 3000.0  # Eval target
    account_busted: bool = False
    _eval_passed_logged: bool = False

    @property
    def drawdown_floor(self) -> float:
        """The equity level at which the account is busted."""
        return self.max_equity_high - self.trailing_threshold

    @property
    def drawdown_remaining(self) -> float:
        """How much equity remains before hitting the drawdown floor."""
        return self.current_equity - self.drawdown_floor

    @property
    def drawdown_used_pct(self) -> float:
        """Percentage of drawdown allowance that has been used (0.0 to 1.0+)."""
        if self.trailing_threshold == 0:
            return 1.0
        return 1.0 - (self.drawdown_remaining / self.trailing_threshold)

    @property
    def realized_profit(self) -> float:
        """How much profit has been made above starting equity."""
        return self.current_equity - self.starting_equity

    @property
    def profit_goal_pct(self) -> float:
        """Progress toward eval profit goal (0.0 to 1.0+)."""
        if self.profit_goal <= 0:
            return 1.0
        return max(0.0, self.realized_profit / self.profit_goal)

    @property
    def eval_passed(self) -> bool:
        """Whether the eval profit goal has been reached."""
        return self.realized_profit >= self.profit_goal

    @property
    def profit_remaining(self) -> float:
        """Dollars remaining to reach the profit goal."""
        return max(0.0, self.profit_goal - self.realized_profit)

    def update_equity(self, equity: float) -> None:
        """Update current equity and ratchet high water mark.

        If equity makes a new high, the drawdown floor moves up.
        If equity drops to or below the floor, the account is busted.
        """
        self.current_equity = equity

        # Ratchet the high water mark upward
        if equity > self.max_equity_high:
            self.max_equity_high = equity
            logger.info(
                "apex_equity_new_high",
                account_id=self.account_id,
                max_equity=self.max_equity_high,
                new_floor=self.drawdown_floor,
            )

        # Check if eval is passed (log once)
        if self.eval_passed and not self._eval_passed_logged:
            self._eval_passed_logged = True
            logger.info(
                "apex_eval_passed",
                account_id=self.account_id,
                profit=self.realized_profit,
                goal=self.profit_goal,
            )

        # Check if account is busted
        if equity <= self.drawdown_floor:
            self.account_busted = True
            logger.critical(
                "apex_account_busted",
                account_id=self.account_id,
                equity=equity,
                floor=self.drawdown_floor,
                max_equity=self.max_equity_high,
            )

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed under Apex drawdown rules.

        Returns:
            (False, reason) if account is busted or within 10% of drawdown limit.
            (True, "") if trading is allowed.
        """
        if self.account_busted:
            return False, (
                f"Apex account {self.account_id} busted: "
                f"equity ${self.current_equity:,.2f} hit floor ${self.drawdown_floor:,.2f}"
            )

        # Warn/block when within 10% of the trailing drawdown limit
        if self.drawdown_used_pct >= 0.90:
            return False, (
                f"Apex drawdown safety: {self.drawdown_used_pct:.1%} of "
                f"${self.trailing_threshold:,.0f} drawdown used "
                f"(${self.drawdown_remaining:,.2f} remaining)"
            )

        return True, ""
