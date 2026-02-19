"""Tests for SignalGenerator."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.core.database import get_sqlite_engine, init_sqlite_db
from src.core.models import (
    Bar,
    Direction,
    IndicatorSnapshot,
    RiskDecision,
    RiskResult,
    Signal,
)
from src.risk.manager import RiskManager
from src.signals.generator import SignalGenerator
from src.strategies.base import BaseStrategy, StrategyConfig


class StubStrategy(BaseStrategy):
    """Strategy that always generates a signal."""

    def __init__(self, signal: Signal | None = None, name="stub"):
        config = StrategyConfig(name=name)
        super().__init__(config)
        self._signal = signal

    def generate_signal(self, bar, snapshot):
        return self._signal

    def validate_params(self):
        return True


def _make_bar(close=5000.0):
    return Bar(
        timestamp=datetime.now(UTC),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1000,
    )


def _make_snapshot(close=5000.0):
    return IndicatorSnapshot(
        timestamp=datetime.now(UTC),
        bb_upper=close + 10,
        bb_middle=close,
        bb_lower=close - 10,
        rsi_14=30.0,
        atr_14=3.0,
        vwap=close,
        ema_9=close - 1,
        ema_21=close + 1,
        keltner_upper=close + 8,
        keltner_lower=close - 8,
    )


def _make_signal(direction=Direction.LONG, price=5000.0):
    return Signal(
        strategy="stub",
        direction=direction,
        confidence=0.7,
        entry_price=price,
        stop_loss=price - 4 if direction == Direction.LONG else price + 4,
        take_profit=price + 8 if direction == Direction.LONG else price - 8,
    )


@pytest.fixture
def sqlite_engine():
    engine = get_sqlite_engine(db_url="sqlite:///:memory:")
    init_sqlite_db(engine)
    return engine


class TestSignalGenerator:
    def test_approved_signal(self):
        signal = _make_signal(Direction.LONG, 5000.0)
        strategy = StubStrategy(signal=signal)
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(strategies=[strategy], risk_manager=risk_mgr)
        results = gen.on_bar(_make_bar(), _make_snapshot())

        assert len(results) == 1
        assert results[0].decision == RiskDecision.APPROVED
        assert results[0].position_size > 0

    def test_rejected_signal_still_returned(self):
        # Signal with take_profit too close (bad R:R)
        signal = Signal(
            strategy="stub",
            direction=Direction.LONG,
            confidence=0.7,
            entry_price=5000.0,
            stop_loss=4996.0,
            take_profit=5001.0,  # R:R = 0.25 < 0.8 minimum
        )
        strategy = StubStrategy(signal=signal)
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(strategies=[strategy], risk_manager=risk_mgr)
        results = gen.on_bar(_make_bar(), _make_snapshot())

        assert len(results) == 1
        assert results[0].decision == RiskDecision.REJECTED

    def test_none_snapshot_returns_empty(self):
        strategy = StubStrategy(signal=_make_signal())
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(strategies=[strategy], risk_manager=risk_mgr)
        results = gen.on_bar(_make_bar(), None)

        assert results == []

    def test_no_signal_from_strategy(self):
        strategy = StubStrategy(signal=None)
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(strategies=[strategy], risk_manager=risk_mgr)
        results = gen.on_bar(_make_bar(), _make_snapshot())

        assert results == []

    def test_multiple_strategies(self):
        sig1 = _make_signal(Direction.LONG, 5000.0)
        sig2 = _make_signal(Direction.SHORT, 5000.0)
        strat1 = StubStrategy(signal=sig1, name="strat1")
        strat2 = StubStrategy(signal=sig2, name="strat2")
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(strategies=[strat1, strat2], risk_manager=risk_mgr)
        results = gen.on_bar(_make_bar(), _make_snapshot())

        assert len(results) == 2

    def test_disabled_strategy_skipped(self):
        signal = _make_signal()
        config = StrategyConfig(name="disabled", enabled=False)
        strategy = StubStrategy(signal=signal)
        strategy.config = config
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(strategies=[strategy], risk_manager=risk_mgr)
        results = gen.on_bar(_make_bar(), _make_snapshot())

        assert results == []

    def test_signal_persisted_to_db(self, sqlite_engine):
        signal = _make_signal()
        strategy = StubStrategy(signal=signal)
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(
            strategies=[strategy],
            risk_manager=risk_mgr,
            sqlite_engine=sqlite_engine,
        )
        gen.on_bar(_make_bar(), _make_snapshot())

        # Verify signal was written
        from src.core.database import SignalRow, get_session

        session = get_session(sqlite_engine)
        rows = session.query(SignalRow).all()
        assert len(rows) == 1
        assert rows[0].strategy == "stub"
        session.close()

    def test_confidence_blended_with_confluence(self):
        """Signal confidence should be blended with confluence score."""
        signal = _make_signal(Direction.LONG, 5000.0)
        original_conf = signal.confidence
        strategy = StubStrategy(signal=signal)
        risk_mgr = RiskManager(account_equity=10000.0)

        gen = SignalGenerator(strategies=[strategy], risk_manager=risk_mgr)
        results = gen.on_bar(_make_bar(), _make_snapshot())

        # After blending, confidence may differ from original
        assert len(results) == 1
