"""Tests for BaseStrategy abstract class."""

from datetime import UTC, datetime

import pytest

from src.core.models import Bar, Direction, IndicatorSnapshot, Signal
from src.strategies.base import BaseStrategy, StrategyConfig


class ConcreteStrategy(BaseStrategy):
    """Minimal concrete implementation for testing."""

    def generate_signal(self, bar, snapshot):
        return Signal(
            strategy=self.name,
            direction=Direction.LONG,
            confidence=0.7,
            entry_price=bar.close,
            stop_loss=bar.close - 4.0,
            reason="test",
        )

    def validate_params(self):
        return True


class NullStrategy(BaseStrategy):
    """Strategy that never generates signals."""

    def generate_signal(self, bar, snapshot):
        return None

    def validate_params(self):
        return True


def _make_bar(close=5000.0):
    return Bar(
        timestamp=datetime.now(UTC),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=100,
    )


def _make_snapshot(close=5000.0):
    return IndicatorSnapshot(
        timestamp=datetime.now(UTC),
        bb_upper=close + 10,
        bb_middle=close,
        bb_lower=close - 10,
        rsi_14=50.0,
        atr_14=3.0,
        ema_9=close - 1,
        ema_21=close + 1,
    )


class TestBaseStrategy:
    def test_name_from_config(self):
        config = StrategyConfig(name="test_strat")
        strat = ConcreteStrategy(config)
        assert strat.name == "test_strat"

    def test_enabled_default_true(self):
        config = StrategyConfig(name="test")
        strat = ConcreteStrategy(config)
        assert strat.enabled is True

    def test_on_bar_returns_none_when_disabled(self):
        config = StrategyConfig(name="test", enabled=False)
        strat = ConcreteStrategy(config)
        result = strat.on_bar(_make_bar(), _make_snapshot())
        assert result is None

    def test_on_bar_returns_none_when_snapshot_is_none(self):
        config = StrategyConfig(name="test")
        strat = ConcreteStrategy(config)
        result = strat.on_bar(_make_bar(), None)
        assert result is None

    def test_on_bar_delegates_to_generate_signal(self):
        config = StrategyConfig(name="test")
        strat = ConcreteStrategy(config)
        bar = _make_bar()
        snapshot = _make_snapshot()
        result = strat.on_bar(bar, snapshot)
        assert result is not None
        assert result.direction == Direction.LONG
        assert result.strategy == "test"

    def test_on_bar_stores_last_snapshot(self):
        config = StrategyConfig(name="test")
        strat = ConcreteStrategy(config)
        snapshot = _make_snapshot()
        strat.on_bar(_make_bar(), snapshot)
        assert strat._last_snapshot is snapshot

    def test_null_strategy_returns_none(self):
        config = StrategyConfig(name="null")
        strat = NullStrategy(config)
        result = strat.on_bar(_make_bar(), _make_snapshot())
        assert result is None

    def test_reset_clears_last_snapshot(self):
        config = StrategyConfig(name="test")
        strat = ConcreteStrategy(config)
        strat.on_bar(_make_bar(), _make_snapshot())
        assert strat._last_snapshot is not None
        strat.reset()
        assert strat._last_snapshot is None

    def test_strategy_config_defaults(self):
        config = StrategyConfig(name="test")
        assert config.enabled is True
        assert config.min_confidence == 0.5
        assert config.params == {}

    def test_strategy_config_custom_params(self):
        config = StrategyConfig(name="test", params={"rsi_oversold": 30})
        assert config.params["rsi_oversold"] == 30
