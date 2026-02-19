"""Order and position lifecycle management.

Tracks open positions, monitors stop-loss and take-profit levels,
handles trailing stops, and coordinates with PaperExecutor for fills.
"""

from __future__ import annotations

from src.config import MES_SPEC
from src.core.logging import get_logger
from src.core.models import (
    Bar,
    Position,
    RiskDecision,
    RiskResult,
    Trade,
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

        position = Position(trade=trade, current_price=trade.entry_price)
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
