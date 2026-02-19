"""Paper trading executor.

Simulates fills with configurable slippage and latency.
Never touches real money -- enforced by paper_mode check.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

from src.config import MES_SPEC, settings
from src.core.logging import get_logger
from src.core.models import Direction, Position, RiskResult, Trade, TradeStatus

logger = get_logger("paper_executor")


class PaperExecutor:
    """Simulates trade execution for paper trading mode."""

    def __init__(
        self,
        slippage_ticks_mean: float = 0.5,
        slippage_ticks_std: float = 0.25,
        fill_probability: float = 1.0,
        tick_size: float = MES_SPEC["tick_size"],
        commission: float = MES_SPEC["commission_per_side"] * 2,
        paper_mode: bool | None = None,
    ) -> None:
        is_paper = paper_mode if paper_mode is not None else settings.trading.paper_mode
        if not is_paper:
            raise RuntimeError("PaperExecutor requires paper_mode=True")
        self.slippage_mean = slippage_ticks_mean
        self.slippage_std = slippage_ticks_std
        self.fill_probability = fill_probability
        self.tick_size = tick_size
        self.commission = commission

    def execute_entry(self, risk_result: RiskResult) -> Trade | None:
        """Simulate a market order fill for an approved signal.

        Returns Trade with simulated fill price, or None if fill skipped.
        """
        if risk_result.signal is None:
            return None

        # Probabilistic fill
        if self.fill_probability < 1.0 and random.random() > self.fill_probability:
            logger.info("fill_skipped", reason="fill_probability")
            return None

        signal = risk_result.signal
        fill_price = self._apply_slippage(
            signal.entry_price, signal.direction, is_entry=True
        )

        trade = Trade(
            strategy=signal.strategy,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            quantity=risk_result.position_size,
            entry_time=datetime.now(UTC),
            status=TradeStatus.OPEN,
            commission=self.commission * risk_result.position_size,
            slippage_ticks=abs(fill_price - signal.entry_price) / self.tick_size,
            signal_confidence=signal.confidence,
        )

        logger.info(
            "paper_entry",
            direction=signal.direction.value,
            entry=fill_price,
            stop=signal.stop_loss,
            target=signal.take_profit,
            qty=risk_result.position_size,
        )

        return trade

    def execute_exit(
        self, position: Position, exit_price: float, reason: str = ""
    ) -> Trade:
        """Close a position at the given price with slippage."""
        trade = position.trade
        fill_price = self._apply_slippage(
            exit_price, trade.direction, is_entry=False
        )

        trade.exit_price = fill_price
        trade.exit_time = datetime.now(UTC)
        trade.status = TradeStatus.CLOSED
        trade.notes = reason
        trade.calculate_pnl()
        trade.calculate_risk_reward()

        logger.info(
            "paper_exit",
            direction=trade.direction.value,
            exit=fill_price,
            pnl=trade.pnl_dollars,
            reason=reason,
        )

        return trade

    def _apply_slippage(
        self, price: float, direction: Direction, is_entry: bool
    ) -> float:
        """Apply random slippage (always adverse).

        Entry: LONG slips UP, SHORT slips DOWN
        Exit: LONG slips DOWN, SHORT slips UP
        """
        slip_ticks = max(0, random.gauss(self.slippage_mean, self.slippage_std))
        slip_price = slip_ticks * self.tick_size

        # Determine direction of slippage (always adverse)
        if is_entry:
            if direction == Direction.LONG:
                price += slip_price  # Worse fill for long entry
            else:
                price -= slip_price  # Worse fill for short entry
        else:
            if direction == Direction.LONG:
                price -= slip_price  # Worse fill for long exit
            else:
                price += slip_price  # Worse fill for short exit

        return self._round_to_tick(price)

    def _round_to_tick(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)
