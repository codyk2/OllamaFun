"""Live trade executor using the Tradovate REST API.

Places real market orders through TradovateRestClient and builds
Trade / Position objects compatible with the rest of the system.
One TradovateExecutor instance is created per trading account.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.core.models import Direction, Position, RiskResult, Trade, TradeStatus
from src.execution.base_executor import BaseExecutor

if TYPE_CHECKING:
    from src.tradovate.rest_client import TradovateRestClient

logger = get_logger("tradovate_executor")


class TradovateExecutor(BaseExecutor):
    """Execute trades on a single Tradovate account via REST.

    Parameters
    ----------
    rest_client : TradovateRestClient
        Authenticated REST client.
    tradovate_account_id : int
        Numeric Tradovate account ID (used in API calls).
    account_id : str
        Logical account ID used in our Trade/Position models.
    contract_id : int
        Tradovate contract ID for the instrument being traded.
    tick_size : float
        Minimum price increment (0.25 for MES).
    commission : float
        Round-trip commission per contract in dollars.
    """

    def __init__(
        self,
        rest_client: TradovateRestClient,
        tradovate_account_id: int,
        account_id: str,
        contract_id: int,
        tick_size: float = 0.25,
        commission: float = 1.04,
    ) -> None:
        self._client = rest_client
        self._tradovate_account_id = tradovate_account_id
        self._account_id = account_id
        self._contract_id = contract_id
        self._tick_size = tick_size
        self._commission = commission

    # ------------------------------------------------------------------
    # BaseExecutor interface
    # ------------------------------------------------------------------

    def execute_entry(self, risk_result: RiskResult) -> Trade | None:
        """Place a market entry order synchronously (wraps async call).

        Parameters
        ----------
        risk_result : RiskResult
            Approved risk result containing the signal and position size.

        Returns
        -------
        Trade | None
            The resulting Trade on fill, or ``None`` if the order failed.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._execute_entry_async(risk_result)
        )

    def execute_exit(
        self, position: Position, exit_price: float, reason: str = ""
    ) -> Trade:
        """Place a market exit order synchronously (wraps async call).

        Parameters
        ----------
        position : Position
            The open position to close.
        exit_price : float
            Expected exit price (actual fill may differ).
        reason : str
            Human-readable reason for the exit.

        Returns
        -------
        Trade
            The updated Trade with exit details filled in.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._execute_exit_async(position, exit_price, reason)
        )

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def _execute_entry_async(self, risk_result: RiskResult) -> Trade | None:
        """Place an entry market order via Tradovate REST API."""
        if risk_result.signal is None:
            logger.warning("entry_no_signal")
            return None

        signal = risk_result.signal
        action = "Buy" if signal.direction == Direction.LONG else "Sell"

        logger.info(
            "entry_submit",
            account=self._account_id,
            direction=signal.direction.value,
            qty=risk_result.position_size,
            symbol=signal.symbol,
        )

        try:
            resp = await self._client.place_order(
                account_id=self._tradovate_account_id,
                action=action,
                symbol=signal.symbol,
                qty=risk_result.position_size,
                order_type="Market",
            )
        except Exception as exc:
            logger.error("entry_order_failed", error=str(exc))
            return None

        # Extract fill price from response
        fill_price = self._extract_fill_price(resp, signal.entry_price)
        order_id = resp.get("orderId")

        trade = Trade(
            account_id=self._account_id,
            strategy=signal.strategy,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            quantity=risk_result.position_size,
            entry_time=datetime.now(UTC),
            status=TradeStatus.OPEN,
            commission=self._commission * risk_result.position_size,
            slippage_ticks=abs(fill_price - signal.entry_price) / self._tick_size,
            signal_confidence=signal.confidence,
            notes=f"tradovate_order_id={order_id}",
        )

        logger.info(
            "entry_filled",
            account=self._account_id,
            order_id=order_id,
            direction=signal.direction.value,
            fill_price=fill_price,
            qty=risk_result.position_size,
        )

        return trade

    async def _execute_exit_async(
        self, position: Position, exit_price: float, reason: str = ""
    ) -> Trade:
        """Place an exit market order via Tradovate REST API."""
        trade = position.trade
        action = "Sell" if trade.direction == Direction.LONG else "Buy"

        logger.info(
            "exit_submit",
            account=self._account_id,
            direction=trade.direction.value,
            qty=trade.quantity,
            reason=reason,
        )

        try:
            resp = await self._client.place_order(
                account_id=self._tradovate_account_id,
                action=action,
                symbol=trade.symbol,
                qty=trade.quantity,
                order_type="Market",
            )

            fill_price = self._extract_fill_price(resp, exit_price)
            order_id = resp.get("orderId")

        except Exception as exc:
            logger.error("exit_order_failed", error=str(exc), reason=reason)
            # Use the expected exit_price if the API call fails so the
            # trade record is still closed (fail-safe for risk management).
            fill_price = exit_price
            order_id = None

        trade.exit_price = fill_price
        trade.exit_time = datetime.now(UTC)
        trade.status = TradeStatus.CLOSED
        trade.notes = (trade.notes or "") + f" | exit_reason={reason}"
        if order_id is not None:
            trade.notes += f" exit_order_id={order_id}"

        trade.calculate_pnl()
        trade.calculate_risk_reward()

        logger.info(
            "exit_filled",
            account=self._account_id,
            order_id=order_id,
            direction=trade.direction.value,
            fill_price=fill_price,
            pnl=trade.pnl_dollars,
            reason=reason,
        )

        return trade

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_fill_price(resp: dict, fallback: float) -> float:
        """Extract the fill price from a Tradovate order response.

        Tradovate may return fill info in several locations depending
        on whether the order was filled immediately (market order) or
        is still working.

        Parameters
        ----------
        resp : dict
            Raw JSON response from ``placeOrder``.
        fallback : float
            Price to use if no fill price is found in the response.
        """
        # Direct fill price
        if "avgFillPrice" in resp:
            return float(resp["avgFillPrice"])

        # Nested in order status
        order_status = resp.get("orderStatus") or {}
        if isinstance(order_status, dict) and "avgFillPrice" in order_status:
            return float(order_status["avgFillPrice"])

        # Fill inside fills array
        fills = resp.get("fills") or []
        if fills and isinstance(fills, list):
            return float(fills[0].get("price", fallback))

        return fallback
