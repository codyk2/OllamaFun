"""Tradovate market data WebSocket provider.

Replaces the old IBProvider with an identical callback interface so that
BarAggregator works unchanged.  Connects to the Tradovate market-data
WebSocket, authenticates, subscribes to 5-second chart bars, and emits
Bar objects through the ``on_bar`` / ``on_tick`` callbacks.

Tradovate MD WebSocket protocol
--------------------------------
- After connecting, the server sends ``o`` (open frame).
- Client authenticates with ``authorize\n<seq>\n\n<json>``.
- Client requests chart data with ``md/getChart\n<seq>\n\n<json>``.
- Data frames arrive as ``a[...]`` JSON arrays.
- Chart bar updates have ``e: "chart"`` with OHLCV data in ``d.bars``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Callable

import websockets
import websockets.asyncio.client

from src.core.logging import get_logger
from src.core.models import Bar

if TYPE_CHECKING:
    from src.tradovate.auth import TradovateAuth

logger = get_logger("tradovate_md")

# Reconnect backoff parameters
_INITIAL_BACKOFF_S: float = 1.0
_MAX_BACKOFF_S: float = 60.0
_BACKOFF_FACTOR: float = 2.0


class TradovateProvider:
    """Real-time market data from Tradovate via WebSocket.

    Drop-in replacement for the old IBProvider -- exposes the same
    ``on_bar`` callback that ``BarAggregator.on_bar`` hooks into.

    Parameters
    ----------
    auth : TradovateAuth
        Token provider.
    config : object
        Must expose ``md_ws_url``.
    on_bar : Callable[[Bar], None] | None
        Called for every completed 5-second chart bar.
    on_tick : Callable | None
        Called for raw tick data (reserved for future use).
    """

    def __init__(
        self,
        auth: TradovateAuth,
        config: object,
        on_bar: Callable[[Bar], None] | None = None,
        on_tick: Callable | None = None,
    ) -> None:
        self._auth = auth
        self._ws_url: str = config.md_ws_url  # type: ignore[attr-defined]
        self.on_bar = on_bar
        self.on_tick = on_tick

        self._ws: websockets.asyncio.client.ClientConnection | None = None
        self._connected = False
        self._seq: int = 0
        self._subscribed_symbol: str | None = None
        self._should_run = False

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
        """Open the market-data WebSocket and authenticate.

        Returns ``True`` on success, ``False`` on failure.
        """
        try:
            self._ws = await websockets.asyncio.client.connect(self._ws_url)

            # Wait for server open frame
            open_frame = await asyncio.wait_for(self._ws.recv(), timeout=10)
            if not isinstance(open_frame, str) or not open_frame.startswith("o"):
                logger.warning("md_unexpected_open_frame", frame=str(open_frame)[:100])

            # Authenticate
            token = await self._auth.get_token()
            auth_msg = self._build_message("authorize", "", {"token": token})
            await self._ws.send(auth_msg)

            # Read auth response
            auth_resp = await asyncio.wait_for(self._ws.recv(), timeout=10)
            logger.info("md_ws_auth_response", resp=str(auth_resp)[:200])

            self._connected = True
            logger.info("md_ws_connected", url=self._ws_url)
            return True

        except Exception as exc:
            logger.error("md_ws_connect_failed", error=str(exc))
            self._connected = False
            return False

    async def subscribe_realtime_bars(self, symbol: str = "MES") -> bool:
        """Request 5-second chart bars for the given symbol.

        Parameters
        ----------
        symbol : str
            Tradovate symbol root (default ``"MES"``).

        Returns ``True`` on success.
        """
        if not self._ws or not self._connected:
            logger.error("subscribe_bars_not_connected")
            return False

        try:
            chart_request = {
                "symbol": symbol,
                "chartDescription": {
                    "underlyingType": "MinuteBar",
                    "elementSize": 5,
                    "elementSizeUnit": "UnderlyingUnits",
                    "withHistogram": False,
                },
                "timeRange": {
                    "asFarAsTimestamp": datetime.now(UTC).isoformat(),
                    "closestTimestamp": datetime.now(UTC).isoformat(),
                },
            }
            msg = self._build_message("md/getChart", "", chart_request)
            await self._ws.send(msg)
            self._subscribed_symbol = symbol
            logger.info("md_ws_chart_subscribed", symbol=symbol)
            return True

        except Exception as exc:
            logger.error("md_ws_subscribe_failed", symbol=symbol, error=str(exc))
            return False

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._should_run = False
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("md_ws_disconnected")

    # ------------------------------------------------------------------
    # Main receive loop
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Receive loop with automatic reconnection and exponential backoff.

        Parses chart data frames and emits ``Bar`` objects through the
        ``on_bar`` callback.  On disconnect, waits with exponential
        backoff and attempts to reconnect + resubscribe.
        """
        self._should_run = True
        backoff = _INITIAL_BACKOFF_S

        while self._should_run:
            # Ensure we are connected
            if not self._connected:
                ok = await self.connect()
                if not ok:
                    logger.warning("md_ws_reconnect_backoff", wait_s=backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * _BACKOFF_FACTOR, _MAX_BACKOFF_S)
                    continue

                # Re-subscribe after reconnect
                if self._subscribed_symbol:
                    await self.subscribe_realtime_bars(self._subscribed_symbol)

                backoff = _INITIAL_BACKOFF_S  # reset on success

            # Read messages
            try:
                async for raw in self._ws:  # type: ignore[union-attr]
                    if not self._should_run:
                        break

                    if not isinstance(raw, str):
                        continue

                    if raw.startswith("h"):
                        # Heartbeat
                        continue

                    if raw.startswith("c"):
                        logger.warning("md_ws_server_close", frame=raw[:200])
                        break

                    if raw.startswith("a"):
                        self._handle_payload(raw[1:])

            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning("md_ws_closed", code=exc.code, reason=exc.reason)
            except Exception as exc:
                logger.error("md_ws_run_error", error=str(exc))
            finally:
                self._connected = False

            if self._should_run:
                logger.info("md_ws_reconnecting", wait_s=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_FACTOR, _MAX_BACKOFF_S)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _build_message(self, endpoint: str, query: str, body: dict) -> str:
        """Build a Tradovate WS protocol message."""
        seq = self._next_seq()
        body_json = json.dumps(body)
        return f"{endpoint}\n{seq}\n{query}\n{body_json}"

    def _handle_payload(self, raw_json: str) -> None:
        """Parse a JSON array payload and extract chart bar data."""
        try:
            messages = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("md_ws_json_parse_error", raw=raw_json[:200])
            return

        if not isinstance(messages, list):
            messages = [messages]

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            event_type = msg.get("e")
            data = msg.get("d")

            if event_type == "chart" and data:
                self._process_chart_data(data)

    def _process_chart_data(self, data: dict) -> None:
        """Extract OHLCV bars from a chart event and emit via on_bar."""
        charts = data.get("charts") or data.get("bars")
        if not charts:
            # Some responses nest bars differently
            bars_list = data.get("bars")
            if bars_list is None:
                # Try top-level bar fields for single-bar updates
                if all(k in data for k in ("open", "high", "low", "close")):
                    bars_list = [data]
                else:
                    return
            charts = bars_list

        if not isinstance(charts, list):
            charts = [charts]

        for bar_data in charts:
            bar = self._parse_bar(bar_data)
            if bar and self.on_bar:
                try:
                    self.on_bar(bar)
                except Exception as exc:
                    logger.error("on_bar_callback_error", error=str(exc))

    def _parse_bar(self, bar_data: dict) -> Bar | None:
        """Convert a Tradovate chart bar dict into a Bar model.

        Tradovate bar fields:
            timestamp (epoch ms or ISO), open, high, low, close,
            upVolume + downVolume (or volume), upTicks + downTicks
        """
        try:
            # Parse timestamp -- Tradovate sends epoch milliseconds or ISO strings
            raw_ts = bar_data.get("timestamp")
            if raw_ts is None:
                ts = datetime.now(UTC)
            elif isinstance(raw_ts, (int, float)):
                ts = datetime.fromtimestamp(raw_ts / 1000.0, tz=UTC)
            else:
                ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))

            # Volume: Tradovate splits into upVolume/downVolume
            up_vol = bar_data.get("upVolume", 0) or 0
            down_vol = bar_data.get("downVolume", 0) or 0
            total_vol = bar_data.get("volume", up_vol + down_vol)

            # Trade count
            up_ticks = bar_data.get("upTicks", 0) or 0
            down_ticks = bar_data.get("downTicks", 0) or 0
            trade_count = bar_data.get("tradeCount") or (up_ticks + down_ticks) or None

            return Bar(
                timestamp=ts,
                symbol=self._subscribed_symbol or "MES",
                open=float(bar_data["open"]),
                high=float(bar_data["high"]),
                low=float(bar_data["low"]),
                close=float(bar_data["close"]),
                volume=int(total_vol),
                trade_count=int(trade_count) if trade_count else None,
            )

        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("bar_parse_error", error=str(exc), data=str(bar_data)[:200])
            return None
