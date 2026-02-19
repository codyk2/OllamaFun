"""Interactive Brokers market data provider using ib_async.

Connects to TWS/IB Gateway, subscribes to MES tick data,
and emits bars via callbacks. Handles auto-reconnect.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable

import pytz
from ib_async import IB, Contract, Future, util

from src.config import settings
from src.core.logging import get_logger
from src.core.models import Bar

logger = get_logger("ib_provider")

CT = pytz.timezone("America/Chicago")


def mes_contract() -> Contract:
    """Create the MES (Micro E-mini S&P 500) continuous futures contract."""
    return Future(
        symbol="MES",
        exchange="CME",
        currency="USD",
        lastTradeDateOrContractMonth="",  # Will be qualified by IB
    )


class IBProvider:
    """Manages IB connection and real-time market data subscription."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        on_bar: Callable[[Bar], None] | None = None,
        on_tick: Callable[[dict], None] | None = None,
    ) -> None:
        self.host = host or settings.ib.host
        self.port = port or settings.ib.port
        self.client_id = client_id or settings.ib.client_id
        self.on_bar = on_bar
        self.on_tick = on_tick

        self.ib = IB()
        self.contract: Contract | None = None
        self._connected = False
        self._reconnect_task: asyncio.Task | None = None
        self._running = False

        # Wire up disconnect handler
        self.ib.disconnectedEvent += self._on_disconnect

    @property
    def connected(self) -> bool:
        return self.ib.isConnected()

    async def connect(self) -> bool:
        """Connect to TWS/IB Gateway."""
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                readonly=True,
            )
            self._connected = True
            logger.info("ib_connected", host=self.host, port=self.port)
            return True
        except Exception as e:
            logger.error("ib_connect_failed", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect from IB."""
        self._running = False
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self.connected:
            self.ib.disconnect()
        self._connected = False
        logger.info("ib_disconnected")

    async def subscribe_realtime_bars(self, symbol: str = "MES") -> bool:
        """Subscribe to 5-second real-time bars for a futures contract."""
        if not self.connected:
            logger.error("ib_not_connected", action="subscribe_bars")
            return False

        try:
            contract = mes_contract()
            qualified = await self.ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.error("ib_qualify_failed", symbol=symbol)
                return False

            self.contract = qualified[0]
            logger.info("ib_contract_qualified", contract=str(self.contract))

            bars = self.ib.reqRealTimeBars(
                self.contract,
                barSize=5,
                whatToShow="TRADES",
                useRTH=False,
            )
            bars.updateEvent += self._on_realtime_bar
            self._running = True
            logger.info("ib_realtime_bars_subscribed", symbol=symbol)
            return True
        except Exception as e:
            logger.error("ib_subscribe_failed", error=str(e))
            return False

    async def request_historical_bars(
        self,
        duration: str = "1 D",
        bar_size: str = "1 min",
        symbol: str = "MES",
    ) -> list[Bar]:
        """Request historical bars from IB."""
        if not self.connected:
            logger.error("ib_not_connected", action="historical_bars")
            return []

        try:
            contract = self.contract or mes_contract()
            if not self.contract:
                qualified = await self.ib.qualifyContractsAsync(contract)
                if not qualified:
                    return []
                contract = qualified[0]

            ib_bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )

            bars = []
            for b in ib_bars:
                bars.append(Bar(
                    timestamp=b.date if isinstance(b.date, datetime) else datetime.fromisoformat(str(b.date)),
                    symbol=symbol,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=int(b.volume),
                ))
            logger.info("ib_historical_received", count=len(bars), duration=duration)
            return bars
        except Exception as e:
            logger.error("ib_historical_failed", error=str(e))
            return []

    async def get_account_summary(self) -> dict:
        """Get account equity and P&L info."""
        if not self.connected:
            return {}

        try:
            summary = self.ib.accountValues()
            result = {}
            for av in summary:
                if av.tag in ("NetLiquidation", "UnrealizedPnL", "RealizedPnL"):
                    result[av.tag] = float(av.value)
            return result
        except Exception as e:
            logger.error("ib_account_summary_failed", error=str(e))
            return {}

    def _on_realtime_bar(self, bars, has_new_bar):
        """Callback when a new 5-second bar arrives."""
        if not has_new_bar or not bars:
            return

        bar = bars[-1]
        try:
            model_bar = Bar(
                timestamp=bar.date if isinstance(bar.date, datetime) else datetime.fromisoformat(str(bar.date)),
                symbol=self.contract.symbol if self.contract else "MES",
                open=bar.open_,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=int(bar.volume),
            )
            if self.on_bar:
                self.on_bar(model_bar)
            if self.on_tick:
                self.on_tick({
                    "price": bar.close,
                    "volume": int(bar.volume),
                    "time": str(bar.date),
                })
        except Exception as e:
            logger.error("ib_bar_callback_error", error=str(e))

    def _on_disconnect(self):
        """Handle IB disconnection."""
        self._connected = False
        logger.warning("ib_disconnected_unexpectedly")
        if self._running:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._reconnect_task = loop.create_task(self._auto_reconnect())

    async def _auto_reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        delays = [1, 2, 4, 8, 16, 30, 60]
        attempt = 0

        while self._running and not self.connected:
            delay = delays[min(attempt, len(delays) - 1)]
            logger.info("ib_reconnecting", attempt=attempt + 1, delay=delay)
            await asyncio.sleep(delay)

            try:
                success = await self.connect()
                if success:
                    logger.info("ib_reconnected", attempt=attempt + 1)
                    if self.contract:
                        await self.subscribe_realtime_bars()
                    return
            except Exception as e:
                logger.error("ib_reconnect_failed", attempt=attempt + 1, error=str(e))

            attempt += 1

        logger.warning("ib_reconnect_gave_up")

    async def run_forever(self) -> None:
        """Keep the IB event loop running."""
        self._running = True
        while self._running:
            if self.connected:
                await asyncio.sleep(1)
                self.ib.sleep(0)  # Process IB events
            else:
                await asyncio.sleep(5)
