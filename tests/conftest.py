"""Shared test fixtures."""

import pytest

from src.core.models import Direction, Signal


@pytest.fixture
def sample_long_signal():
    """A valid long signal for testing."""
    return Signal(
        strategy="mean_reversion",
        symbol="MES",
        direction=Direction.LONG,
        confidence=0.7,
        entry_price=5000.00,
        stop_loss=4996.00,  # 16 ticks below entry
        take_profit=5004.75,  # 19 ticks above entry (~1.2:1 R:R)
        reason="BB lower touch + RSI oversold",
    )


@pytest.fixture
def sample_short_signal():
    """A valid short signal for testing."""
    return Signal(
        strategy="mean_reversion",
        symbol="MES",
        direction=Direction.SHORT,
        confidence=0.65,
        entry_price=5020.00,
        stop_loss=5024.00,  # 16 ticks above entry
        take_profit=5015.25,  # 19 ticks below entry (~1.2:1 R:R)
        reason="BB upper touch + RSI overbought",
    )
