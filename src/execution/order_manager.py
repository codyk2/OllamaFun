"""Order and position lifecycle management.

Tracks open positions, monitors stop-loss and take-profit levels,
handles trailing stops, and coordinates with PaperExecutor for fills.
"""

from __future__ import annotations

from src.config import MES_SPEC
from src.core.logging import get_logger
from src.core.models import (
    Bar,
    Direction,
    Position,
    RiskDecision,
    RiskResult,
    Trade,
    TradeStatus,
)
from src.execution.paper_executor import PaperExecutor
from src.journal.recorder import TradeRecorder
from src.risk.manager import RiskManager
from src.risk.stop_loss import update_trailing_stop

logger = get_logger("order_manager")


class OrderManager:
    """Manages open positions and order lifecycle."""

    def __init__(
        self,
        executor: PaperExecutor,
        risk_manager: RiskManager,
        recorder: TradeRecorder | None = None,
        atr_for_trailing: float = 3.0,
    ) -> None:
        self.executor = executor
        self.risk_manager = risk_manager
        self.recorder = recorder
        self.atr_for_trailing = atr_for_trailing
        self.open_positions: list[Position] = []

    def process_signals(self, risk_results: list[RiskResult]) -> list[Trade]:
        """Process approved risk results into open trades."""
        new_trades: list[Trade] = []

        for result in risk_results:
            if result.decision != RiskDecision.APPROVED:
                continue

            position = self._open_position(result)
            if position is not None:
                new_trades.append(position.trade)

        return new_trades

    def on_bar(self, bar: Bar) -> list[Trade]:
        """Check all open positions against new bar for exits.

        Updates prices, checks stop-loss, take-profit, trailing stops.
        Returns list of closed trades.
        """
        closed_trades: list[Trade] = []
        remaining: list[Position] = []

        for position in self.open_positions:
            position.update_price(bar.close)

            # Update trailing stop
            self._update_trailing_stop(position, bar)

            # Check exit conditions
            if position.should_stop_out():
                stop = position.trailing_stop or position.trade.stop_loss
                trade = self._close_position(position, stop, "stop_loss")
                closed_trades.append(trade)
            elif self._should_scale_out(position, bar):
                self._execute_scale_out(position, bar)
                remaining.append(position)
            elif position.should_take_profit():
                trade = self._close_position(
                    position, position.trade.take_profit, "take_profit"
                )
                closed_trades.append(trade)
            else:
                remaining.append(position)

        self.open_positions = remaining
        return closed_trades

    def force_close_all(self, current_price: float) -> list[Trade]:
        """Force-close all open positions at the given price."""
        closed: list[Trade] = []
        for position in self.open_positions:
            trade = self._close_position(position, current_price, "force_close")
            closed.append(trade)
        self.open_positions = []
        return closed

    def get_total_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all open positions."""
        return sum(p.unrealized_pnl for p in self.open_positions)

    def _open_position(self, risk_result: RiskResult) -> Position | None:
        """Execute entry and create Position."""
        trade = self.executor.execute_entry(risk_result)
        if trade is None:
            return None

        position = Position(
            trade=trade,
            current_price=trade.entry_price,
            original_quantity=trade.quantity,
        )
        self.open_positions.append(position)
        self.risk_manager.record_position_opened()

        logger.info(
            "position_opened",
            direction=trade.direction.value,
            entry=trade.entry_price,
            qty=trade.quantity,
        )

        return position

    def _close_position(
        self, position: Position, exit_price: float, reason: str
    ) -> Trade:
        """Execute exit, update risk manager, record trade."""
        trade = self.executor.execute_exit(position, exit_price, reason)

        self.risk_manager.record_position_closed(trade.pnl_dollars or 0.0)

        if self.recorder:
            self.recorder.record_trade(trade)

        logger.info(
            "position_closed",
            direction=trade.direction.value,
            pnl=trade.pnl_dollars,
            reason=reason,
        )

        return trade

    def _update_trailing_stop(self, position: Position, bar: Bar) -> None:
        """Update trailing stop using ATR."""
        current_stop = position.trailing_stop or position.trade.stop_loss
        new_stop, activated = update_trailing_stop(
            entry_price=position.trade.entry_price,
            current_price=bar.close,
            current_stop=current_stop,
            direction=position.trade.direction,
            atr=self.atr_for_trailing,
        )

        if activated:
            position.trailing_stop = new_stop
            position.trailing_activated = True

    def _should_scale_out(self, position: Position, bar: Bar) -> bool:
        """Check if position should scale out at primary target.

        Scale-out conditions:
        - Signal had take_profit_primary set
        - Position quantity > 1 (can't split a 1-lot)
        - Scale-out hasn't been done yet
        - Price has reached primary target
        """
        if position.scale_out_done:
            return False
        if position.trade.quantity <= 1:
            return False

        signal = getattr(position.trade, '_signal', None)
        # Access primary target from the trade's original signal via the take_profit_primary
        # We store it on the trade as a note when the signal has dual targets
        primary = self._get_primary_target(position)
        if primary is None:
            return False

        if position.trade.direction == Direction.LONG:
            return bar.close >= primary
        return bar.close <= primary

    def _get_primary_target(self, position: Position) -> float | None:
        """Get the primary take-profit target from the trade's notes or take_profit field.

        When scale-out is available, take_profit is the primary and
        take_profit on the trade may be updated to the secondary after scale-out.
        """
        # For scale-out, the trade's take_profit starts as the primary target.
        # If the position hasn't scaled out yet, take_profit IS the primary.
        if not position.scale_out_done and position.trade.quantity > 1:
            return position.trade.take_profit
        return None

    def _execute_scale_out(self, position: Position, bar: Bar) -> None:
        """Close half the position at primary target, move stop to breakeven."""
        primary = position.trade.take_profit
        if primary is None:
            return

        scale_qty = position.trade.quantity // 2
        if scale_qty <= 0:
            return

        # Reduce position quantity
        position.trade.quantity -= scale_qty
        position.scale_out_done = True

        # Move stop to breakeven (entry price)
        position.trailing_stop = position.trade.entry_price
        position.trailing_activated = True

        # Record partial P&L
        if position.trade.direction == Direction.LONG:
            pnl_ticks = (primary - position.trade.entry_price) / 0.25
        else:
            pnl_ticks = (position.trade.entry_price - primary) / 0.25
        pnl_dollars = pnl_ticks * 1.25 * scale_qty

        self.risk_manager.daily_tracker.record_trade_closed(pnl_dollars)

        logger.info(
            "scale_out",
            direction=position.trade.direction.value,
            qty_closed=scale_qty,
            qty_remaining=position.trade.quantity,
            price=primary,
            pnl=pnl_dollars,
        )
