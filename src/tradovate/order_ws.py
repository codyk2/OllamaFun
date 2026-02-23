"""Tradovate order WebSocket for real-time fill and order notifications.

Connects to the Tradovate order/execution WebSocket endpoint,
authenticates with a bearer token, and streams fill events to
a callback function.

Tradovate WS protocol
---------------------
- After connecting, the server sends an ``o`` (open) frame.
- Client sends ``authorize\n<seq>\n\n<json>`` to authenticate.
- Client sends ``user/syncrequest\n<seq>\n\n<json>`` to subscribe.
- Inbound frames are prefixed with a single character:
    ``a`` = JSON array payload (normal messages)
    ``h`` = heartbeat
    ``o`` = open
    ``c`` = close
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Callable

import websockets
import websockets.asyncio.client

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.tradovate.auth import TradovateAuth

logger = get_logger("tradovate_order_ws")


class TradovateOrderWS:
    """Real-time order/fill notifications via Tradovate WebSocket.

    Parameters
    ----------
    auth : TradovateAuth
        Token provider.
    config : object
        Must expose ``order_ws_url``.
    """

    def __init__(self, auth: TradovateAuth, config: object) -> None:
        self._auth = auth
        self._ws_url: str = config.order_ws_url  # type: ignore[attr-defined]
        self._ws: websockets.asyncio.client.ClientConnection | None = None
        self._connected = False
        self._seq: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Open the WebSocket and authenticate.

        Returns ``True`` on success, ``False`` on failure.
        """
        try:
            self._ws = await websockets.asyncio.client.connect(self._ws_url)

            # Wait for the server "open" frame
            open_frame = await asyncio.wait_for(self._ws.recv(), timeout=10)
            if not isinstance(open_frame, str) or not open_frame.startswith("o"):
                logger.warning("unexpected_open_frame", frame=str(open_frame)[:100])

            # Authenticate
            token = await self._auth.get_token()
            auth_msg = self._build_message("authorize", "", {"token": token})
            await self._ws.send(auth_msg)

            # Read auth response
            auth_resp = await asyncio.wait_for(self._ws.recv(), timeout=10)
            logger.info("order_ws_auth_response", resp=str(auth_resp)[:200])

            self._connected = True
            logger.info("order_ws_connected", url=self._ws_url)
            return True

        except Exception as exc:
            logger.error("order_ws_connect_failed", error=str(exc))
            self._connected = False
            return False

    async def subscribe_orders(self) -> bool:
        """Subscribe to real-time order/fill updates via user/syncRequest.

        Returns ``True`` on success.
        """
        if not self._ws or not self._connected:
            logger.error("subscribe_orders_not_connected")
            return False

        try:
            sync_msg = self._build_message(
                "user/syncrequest",
                "",
                {"users": [self._auth._config.username]},  # type: ignore[attr-defined]
            )
            await self._ws.send(sync_msg)
            logger.info("order_ws_subscribed")
            return True
        except Exception as exc:
            logger.error("order_ws_subscribe_failed", error=str(exc))
            return False

    async def run_forever(self, on_fill: Callable) -> None:
        """Receive loop: parse messages and dispatch fill events.

        Parameters
        ----------
        on_fill : Callable
            Called with the parsed fill dict whenever a fill event arrives.
            Signature: ``on_fill(fill_data: dict) -> None``
        """
        if not self._ws:
            logger.error("run_forever_not_connected")
            return

        logger.info("order_ws_run_forever_start")

        try:
            async for raw in self._ws:
                if not isinstance(raw, str):
                    continue

                if raw.startswith("h"):
                    # Heartbeat -- keep connection alive
                    continue

                if raw.startswith("c"):
                    logger.warning("order_ws_server_close", frame=raw[:200])
                    break

                if raw.startswith("a"):
                    await self._handle_payload(raw[1:], on_fill)

        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning("order_ws_closed", code=exc.code, reason=exc.reason)
        except Exception as exc:
            logger.error("order_ws_run_error", error=str(exc))
        finally:
            self._connected = False

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("order_ws_disconnected")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _build_message(self, endpoint: str, query: str, body: dict) -> str:
        """Build a Tradovate WS protocol message.

        Format: ``endpoint\\nseq\\nquery\\nbody_json``
        """
        seq = self._next_seq()
        body_json = json.dumps(body)
        return f"{endpoint}\n{seq}\n{query}\n{body_json}"

    async def _handle_payload(self, raw_json: str, on_fill: Callable) -> None:
        """Parse a JSON array payload and dispatch fill events."""
        try:
            messages = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("order_ws_json_parse_error", raw=raw_json[:200])
            return

        if not isinstance(messages, list):
            messages = [messages]

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            event_type = msg.get("e")
            data = msg.get("d")

            if event_type == "fill" and data:
                logger.info(
                    "fill_received",
                    order_id=data.get("orderId"),
                    price=data.get("price"),
                    qty=data.get("qty"),
                )
                try:
                    on_fill(data)
                except Exception as exc:
                    logger.error("on_fill_callback_error", error=str(exc))

            elif event_type == "order":
                logger.debug(
                    "order_event",
                    order_id=data.get("id") if data else None,
                    status=data.get("ordStatus") if data else None,
                )
