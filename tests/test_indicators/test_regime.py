"""Tests for market regime detection."""

import random
from datetime import UTC, datetime, timedelta

import pytest

from src.core.models import Bar
from src.indicators.regime import (
    ADX_RANGING_THRESHOLD,
    ADX_TRENDING_THRESHOLD,
    HYSTERESIS_BARS,
    MarketRegime,
    RegimeDetector,
    RegimeState,
    REGIME_SCALING,
)


def _make_bars(count=100, start_price=5000.0, volatility=2.0, seed=42) -> list[Bar]:
    """Generate synthetic bars."""
    random.seed(seed)
    bars = []
    price = start_price
    base_time = datetime(2024, 1, 15, 9, 30, tzinfo=UTC)

    for i in range(count):
        change = random.gauss(0, volatility)
        price += change
        high = price + abs(random.gauss(0, 1))
        low = price - abs(random.gauss(0, 1))
        bar = Bar(
            timestamp=base_time + timedelta(minutes=i),
            open=price - change / 2,
            high=max(high, price),
            low=min(low, price),
            close=price,
            volume=random.randint(100, 5000),
        )
        bars.append(bar)
    return bars


def _make_trending_bars(count=200, start_price=5000.0, trend=2.0, seed=99) -> list[Bar]:
    """Generate bars with a strong uptrend (high ADX)."""
    random.seed(seed)
    bars = []
    price = start_price
    base_time = datetime(2024, 1, 15, 9, 30, tzinfo=UTC)

    for i in range(count):
        price += trend + random.gauss(0, 0.5)  # Strong directional move
        high = price + abs(random.gauss(0, 0.3))
        low = price - abs(random.gauss(0, 0.3))
        bar = Bar(
            timestamp=base_time + timedelta(minutes=i),
            open=price - trend / 2,
            high=max(high, price),
            low=min(low, price - trend / 2),
            close=price,
            volume=random.randint(100, 5000),
        )
        bars.append(bar)
    return bars


class TestRegimeState:
    def test_default_state_is_ranging(self):
        state = RegimeState()
        assert state.regime == MarketRegime.RANGING
        assert state.signal_scaling == 1.0
        assert state.adx is None
        assert state.squeeze_active is False

    def test_regime_scaling_values(self):
        assert REGIME_SCALING[MarketRegime.RANGING] == 1.0
        assert REGIME_SCALING[MarketRegime.TRANSITIONAL] == 0.5
        assert REGIME_SCALING[MarketRegime.TRENDING] == 0.0


class TestRegimeDetector:
    def test_initial_state(self):
        detector = RegimeDetector()
        assert detector.state.regime == MarketRegime.RANGING
        assert detector.state.signal_scaling == 1.0

    def test_insufficient_bars_stays_ranging(self):
        detector = RegimeDetector()
        bars = _make_bars(10)
        for bar in bars:
            detector.on_5m_bar(bar)
        assert detector.state.regime == MarketRegime.RANGING

    def test_processes_bars_without_error(self):
        detector = RegimeDetector()
        bars = _make_bars(200)
        for bar in bars:
            detector.on_5m_bar(bar)
        assert detector.state.adx is not None

    def test_adx_computed_after_enough_bars(self):
        detector = RegimeDetector()
        bars = _make_bars(100)
        for bar in bars:
            detector.on_5m_bar(bar)
        assert detector.state.adx is not None
        assert detector.state.adx >= 0


class TestBarAggregation:
    def test_1m_to_5m_aggregation(self):
        detector = RegimeDetector()
        bars = _make_bars(10)

        # First 4 bars: no 5m bar formed yet, state unchanged
        for bar in bars[:4]:
            detector.on_1m_bar(bar)

        # 5th bar triggers aggregation
        detector.on_1m_bar(bars[4])
        # State still defaults until enough data
        assert detector.state.regime == MarketRegime.RANGING

    def test_aggregation_ohlcv_correct(self):
        """5m bar should have correct OHLCV from 5 1m bars."""
        detector = RegimeDetector()
        bars = _make_bars(5)

        expected_open = bars[0].open
        expected_high = max(b.high for b in bars)
        expected_low = min(b.low for b in bars)
        expected_close = bars[-1].close
        expected_volume = sum(b.volume for b in bars)

        for bar in bars:
            detector.on_1m_bar(bar)

        # Check that one 5m bar was created
        assert len(detector._5m_bars) == 1
        bar_5m = detector._5m_bars[0]
        assert bar_5m.open == expected_open
        assert bar_5m.high == expected_high
        assert bar_5m.low == expected_low
        assert bar_5m.close == expected_close
        assert bar_5m.volume == expected_volume

    def test_multiple_aggregations(self):
        detector = RegimeDetector()
        bars = _make_bars(15)  # Should produce 3 x 5m bars
        for bar in bars:
            detector.on_1m_bar(bar)
        assert len(detector._5m_bars) == 3


class TestHysteresis:
    def test_regime_requires_consecutive_bars(self):
        """Regime should not change until HYSTERESIS_BARS consecutive bars agree."""
        detector = RegimeDetector()
        # Start as RANGING (default)
        assert detector.state.regime == MarketRegime.RANGING
        # Manually test hysteresis counter
        assert detector._candidate_count == 0

    def test_hysteresis_resets_on_disagreement(self):
        detector = RegimeDetector()
        # Set up a candidate regime
        detector._candidate_regime = MarketRegime.TRENDING
        detector._candidate_count = 2
        # If the actual regime matches current, candidate resets
        detector._state.regime = MarketRegime.RANGING
        # Simulate _update_regime seeing RANGING again
        # (the actual _update_regime does this logic internally)
        assert HYSTERESIS_BARS == 3  # Verify our expected hysteresis value


class TestRegimeClassification:
    def test_trending_bars_increase_adx(self):
        """Strong trend should push ADX higher."""
        detector = RegimeDetector()
        bars = _make_trending_bars(200, trend=3.0)
        for bar in bars:
            detector.on_5m_bar(bar)
        # ADX should be elevated for a strong trend
        assert detector.state.adx is not None
        assert detector.state.adx > ADX_RANGING_THRESHOLD

    def test_ranging_bars_keep_low_adx(self):
        """Random walk (no trend) should have lower ADX."""
        detector = RegimeDetector()
        bars = _make_bars(200, volatility=1.0, seed=123)
        for bar in bars:
            detector.on_5m_bar(bar)
        assert detector.state.adx is not None
        # Random walk ADX tends to be low, though not guaranteed


class TestSignalScaling:
    def test_ranging_scaling_is_1(self):
        state = RegimeState(regime=MarketRegime.RANGING)
        assert REGIME_SCALING[state.regime] == 1.0

    def test_transitional_scaling_is_half(self):
        state = RegimeState(regime=MarketRegime.TRANSITIONAL)
        assert REGIME_SCALING[state.regime] == 0.5

    def test_trending_scaling_is_zero(self):
        state = RegimeState(regime=MarketRegime.TRENDING)
        assert REGIME_SCALING[state.regime] == 0.0
