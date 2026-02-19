"""Auto-reconnection logic with exponential backoff.

Used primarily for IB connection recovery.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from src.core.logging import get_logger

logger = get_logger("reconnect")


class ReconnectManager:
    """Manages reconnection with exponential backoff."""

    def __init__(
        self,
        connect_fn: Callable,
        max_delay: float = 60.0,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_attempts: int = 0,  # 0 = unlimited
        on_reconnected: Callable | None = None,
        on_give_up: Callable | None = None,
    ) -> None:
        self.connect_fn = connect_fn
        self.max_delay = max_delay
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_attempts = max_attempts
        self.on_reconnected = on_reconnected
        self.on_give_up = on_give_up

        self._attempts = 0
        self._running = False

    @property
    def current_delay(self) -> float:
        """Calculate current backoff delay."""
        delay = self.initial_delay * (self.backoff_factor ** self._attempts)
        return min(delay, self.max_delay)

    async def start(self) -> bool:
        """Begin reconnection loop. Returns True if reconnected."""
        self._running = True
        self._attempts = 0

        while self._running:
            if self.max_attempts > 0 and self._attempts >= self.max_attempts:
                logger.warning("reconnect_max_attempts", attempts=self._attempts)
                if self.on_give_up:
                    self.on_give_up()
                return False

            delay = self.current_delay
            logger.info(
                "reconnect_attempt",
                attempt=self._attempts + 1,
                delay=delay,
            )

            await asyncio.sleep(delay)

            try:
                result = await self.connect_fn()
                if result:
                    logger.info("reconnect_success", attempts=self._attempts + 1)
                    self._running = False
                    if self.on_reconnected:
                        self.on_reconnected()
                    return True
            except Exception as e:
                logger.error(
                    "reconnect_failed",
                    attempt=self._attempts + 1,
                    error=str(e),
                )

            self._attempts += 1

        return False

    def stop(self) -> None:
        """Stop the reconnection loop."""
        self._running = False

    def reset(self) -> None:
        """Reset attempt counter."""
        self._attempts = 0
