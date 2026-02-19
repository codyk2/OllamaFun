"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.core.models import Bar, IndicatorSnapshot, Signal


@dataclass
class StrategyConfig:
    """Base configuration for any strategy."""

    name: str
    enabled: bool = True
    min_confidence: float = 0.5
    params: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract strategy interface.

    Strategies receive completed bars with indicator snapshots
    and emit Signal objects when entry conditions are met.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.logger = get_logger(f"strategy.{config.name}")
        self._last_snapshot: IndicatorSnapshot | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def on_bar(self, bar: Bar, snapshot: IndicatorSnapshot | None) -> Signal | None:
        """Process a new bar and optionally emit a signal.

        Returns None if no trade opportunity or strategy is disabled.
        """
        if not self.enabled:
            return None
        if snapshot is None:
            return None

        self._last_snapshot = snapshot
        return self.generate_signal(bar, snapshot)

    @abstractmethod
    def generate_signal(self, bar: Bar, snapshot: IndicatorSnapshot) -> Signal | None:
        """Core strategy logic. Override in subclasses."""
        ...

    @abstractmethod
    def validate_params(self) -> bool:
        """Validate strategy parameters."""
        ...

    def reset(self) -> None:
        """Reset strategy state (e.g., at session boundary)."""
        self._last_snapshot = None
