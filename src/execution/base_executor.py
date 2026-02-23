"""Abstract executor interface for trade execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import Position, RiskResult, Trade


class BaseExecutor(ABC):
    """Abstract executor interface. Implemented by PaperExecutor and TradovateExecutor."""

    @abstractmethod
    def execute_entry(self, risk_result: RiskResult) -> Trade | None:
        """Execute an entry order. Returns Trade or None if fill failed."""
        ...

    @abstractmethod
    def execute_exit(
        self, position: Position, exit_price: float, reason: str = ""
    ) -> Trade:
        """Close a position. Returns the closed Trade."""
        ...
