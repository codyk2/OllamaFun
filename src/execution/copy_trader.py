"""Copy trade manager for multi-account signal execution.

Orchestrates signal processing and position management across multiple
trading accounts, each with its own OrderManager instance. This is the
core of the prop firm copy trading system.
"""

from __future__ import annotations

from src.core.logging import get_logger
from src.core.models import Bar, RiskResult, Trade
from src.execution.order_manager import OrderManager

logger = get_logger("copy_trader")


class CopyTradeManager:
    """Orchestrates copy trading across multiple accounts.

    Each account has its own OrderManager that handles position sizing,
    risk management, and execution independently. The CopyTradeManager
    fans out signals to each account and collects results.

    Args:
        account_managers: Mapping of account_id to its dedicated OrderManager.
    """

    def __init__(self, account_managers: dict[str, OrderManager]) -> None:
        self.account_managers = account_managers
        logger.info(
            "copy_trader_initialized",
            accounts=list(account_managers.keys()),
            count=len(account_managers),
        )

    def process_signals(
        self, risk_results_by_account: dict[str, list[RiskResult]]
    ) -> dict[str, list[Trade]]:
        """Execute approved signals on each account's OrderManager.

        Args:
            risk_results_by_account: Mapping of account_id to the list of
                RiskResult objects that have been evaluated for that account.

        Returns:
            Mapping of account_id to list of newly opened Trade objects.
        """
        trades_by_account: dict[str, list[Trade]] = {}

        for account_id, risk_results in risk_results_by_account.items():
            manager = self.account_managers.get(account_id)
            if manager is None:
                logger.warning(
                    "unknown_account_in_signals",
                    account=account_id,
                )
                continue

            try:
                trades = manager.process_signals(risk_results)
                trades_by_account[account_id] = trades

                if trades:
                    logger.info(
                        "signals_executed",
                        account=account_id,
                        trade_count=len(trades),
                    )
            except Exception:
                logger.exception(
                    "signal_processing_failed",
                    account=account_id,
                )
                trades_by_account[account_id] = []

        return trades_by_account

    def on_bar(self, bar: Bar) -> dict[str, list[Trade]]:
        """Check all accounts' positions for exits on a new bar.

        Forwards the bar to each OrderManager so it can evaluate
        stop-loss, take-profit, and trailing stop conditions.

        Args:
            bar: The latest OHLCV bar.

        Returns:
            Mapping of account_id to list of Trade objects closed on this bar.
        """
        closed_by_account: dict[str, list[Trade]] = {}

        for account_id, manager in self.account_managers.items():
            try:
                closed = manager.on_bar(bar)
                closed_by_account[account_id] = closed

                if closed:
                    logger.info(
                        "positions_closed_on_bar",
                        account=account_id,
                        closed_count=len(closed),
                        bar_close=bar.close,
                    )
            except Exception:
                logger.exception(
                    "on_bar_failed",
                    account=account_id,
                )
                closed_by_account[account_id] = []

        return closed_by_account

    def force_close_all(self, current_price: float) -> dict[str, list[Trade]]:
        """Force close all positions on all accounts.

        Used for end-of-day liquidation, emergency shutdown, or news
        blackout events.

        Args:
            current_price: The price at which to close all positions.

        Returns:
            Mapping of account_id to list of force-closed Trade objects.
        """
        closed_by_account: dict[str, list[Trade]] = {}

        logger.warning(
            "force_close_all_triggered",
            price=current_price,
            account_count=len(self.account_managers),
        )

        for account_id, manager in self.account_managers.items():
            try:
                closed = manager.force_close_all(current_price)
                closed_by_account[account_id] = closed

                if closed:
                    logger.info(
                        "force_closed",
                        account=account_id,
                        closed_count=len(closed),
                        price=current_price,
                    )
            except Exception:
                logger.exception(
                    "force_close_failed",
                    account=account_id,
                )
                closed_by_account[account_id] = []

        return closed_by_account

    def get_total_unrealized_pnl(self) -> dict[str, float]:
        """Get per-account unrealized P&L across all open positions.

        Returns:
            Mapping of account_id to the total unrealized P&L for that account.
        """
        pnl_by_account: dict[str, float] = {}

        for account_id, manager in self.account_managers.items():
            try:
                pnl_by_account[account_id] = manager.get_total_unrealized_pnl()
            except Exception:
                logger.exception(
                    "pnl_calculation_failed",
                    account=account_id,
                )
                pnl_by_account[account_id] = 0.0

        return pnl_by_account
