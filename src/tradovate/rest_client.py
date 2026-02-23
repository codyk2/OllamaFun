"""Tradovate REST API client for account, order, and contract operations.

All methods are async and inject the bearer token from TradovateAuth
into every request automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.tradovate.auth import TradovateAuth

logger = get_logger("tradovate_rest")


class TradovateRestClient:
    """Async REST client for the Tradovate HTTP API.

    Parameters
    ----------
    auth : TradovateAuth
        Token provider -- ``get_token()`` is called before each request.
    config : object
        Must expose ``base_url`` (e.g. ``https://demo.tradovateapi.com/v1``).
    """

    def __init__(self, auth: TradovateAuth, config: object) -> None:
        self._auth = auth
        self._base_url: str = config.base_url  # type: ignore[attr-defined]
        self._client = httpx.AsyncClient(timeout=30.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _headers(self) -> dict[str, str]:
        token = await self._auth.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _get(self, path: str, params: dict | None = None) -> list | dict:
        url = f"{self._base_url}{path}"
        headers = await self._headers()
        resp = await self._client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, payload: dict | None = None) -> dict:
        url = f"{self._base_url}{path}"
        headers = await self._headers()
        resp = await self._client.post(url, headers=headers, json=payload or {})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def list_accounts(self) -> list[dict]:
        """GET /account/list -- return all accounts visible to the user."""
        result = await self._get("/account/list")
        logger.info("accounts_listed", count=len(result))
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self, account_id: int) -> list[dict]:
        """GET /position/list -- return positions filtered by account.

        Parameters
        ----------
        account_id : int
            Tradovate numeric account ID.
        """
        all_positions: list[dict] = await self._get("/position/list")  # type: ignore[assignment]
        filtered = [p for p in all_positions if p.get("accountId") == account_id]
        logger.info("positions_fetched", account_id=account_id, count=len(filtered))
        return filtered

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        account_id: int,
        action: str,
        symbol: str,
        qty: int,
        order_type: str = "Market",
        price: float | None = None,
    ) -> dict:
        """POST /order/placeOrder -- submit a new order.

        Parameters
        ----------
        account_id : int
            Tradovate numeric account ID.
        action : str
            ``"Buy"`` or ``"Sell"``.
        symbol : str
            Contract symbol (e.g. ``"MESM5"``).
        qty : int
            Number of contracts.
        order_type : str
            ``"Market"``, ``"Limit"``, ``"Stop"``, etc.
        price : float | None
            Required for Limit / StopLimit orders; ignored for Market.
        """
        payload: dict = {
            "accountSpec": account_id,
            "accountId": account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": qty,
            "orderType": order_type,
            "isAutomated": True,
        }
        if price is not None:
            payload["price"] = price

        logger.info(
            "order_submit",
            account_id=account_id,
            action=action,
            symbol=symbol,
            qty=qty,
            order_type=order_type,
            price=price,
        )

        result = await self._post("/order/placeOrder", payload)

        logger.info(
            "order_response",
            order_id=result.get("orderId"),
            status=result.get("orderStatus"),
        )
        return result

    async def cancel_order(self, order_id: int) -> dict:
        """POST /order/cancelOrder -- cancel an open order.

        Parameters
        ----------
        order_id : int
            Tradovate order ID to cancel.
        """
        payload = {"orderId": order_id}
        logger.info("order_cancel", order_id=order_id)
        result = await self._post("/order/cancelOrder", payload)
        logger.info("order_cancel_response", result=result)
        return result

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    async def get_contract(self, symbol: str) -> dict:
        """GET /contract/find?name={symbol} -- look up contract details.

        Parameters
        ----------
        symbol : str
            Contract name (e.g. ``"MESM5"``).
        """
        result = await self._get("/contract/find", params={"name": symbol})
        logger.info("contract_found", symbol=symbol, contract_id=result.get("id"))
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        logger.info("rest_client_closed")
