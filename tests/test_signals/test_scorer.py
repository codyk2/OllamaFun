"""Tests for signal confluence scorer."""

from datetime import UTC, datetime

import pytest

from src.core.models import Direction, IndicatorSnapshot, Signal
from src.signals.scorer import (
    adjust_confidence_for_time_of_day,
    adjust_confidence_for_volatility,
    score_confluence,
)


def _make_signal(direction=Direction.LONG, price=5000.0, confidence=0.7):
    return Signal(
        strategy="test",
        direction=direction,
        confidence=confidence,
        entry_price=price,
        stop_loss=price - 4 if direction == Direction.LONG else price + 4,
    )


def _make_snapshot(**kwargs):
    defaults = dict(
        timestamp=datetime.now(UTC),
        bb_upper=5010.0,
        bb_middle=5000.0,
        bb_lower=4990.0,
        rsi_14=50.0,
        atr_14=3.0,
        ema_9=4999.0,
        ema_21=5001.0,
        vwap=5000.0,
        keltner_upper=5008.0,
        keltner_middle=5000.0,
        keltner_lower=4992.0,
    )
    defaults.update(kwargs)
    return IndicatorSnapshot(**defaults)


class TestScoreConfluence:
    def test_perfect_long_confluence(self):
        """Price at lower BB, RSI oversold, below VWAP, EMA9<EMA21, inside Keltner."""
        signal = _make_signal(Direction.LONG, price=4990.0)
        snapshot = _make_snapshot(rsi_14=30.0, vwap=5000.0, ema_9=4998.0, ema_21=5002.0)
        score = score_confluence(signal, snapshot)
        assert score >= 0.8  # Should be high

    def test_perfect_short_confluence(self):
        signal = _make_signal(Direction.SHORT, price=5010.0)
        snapshot = _make_snapshot(rsi_14=70.0, vwap=5000.0, ema_9=5002.0, ema_21=4998.0)
        score = score_confluence(signal, snapshot)
        assert score >= 0.8

    def test_no_confluence_long(self):
        """Price above middle BB, RSI neutral, above VWAP."""
        signal = _make_signal(Direction.LONG, price=5005.0)
        snapshot = _make_snapshot(rsi_14=55.0, vwap=5000.0, ema_9=5002.0, ema_21=4998.0)
        score = score_confluence(signal, snapshot)
        assert score < 0.5

    def test_score_range_0_to_1(self):
        signal = _make_signal(Direction.LONG, price=4990.0)
        snapshot = _make_snapshot(rsi_14=25.0)
        score = score_confluence(signal, snapshot)
        assert 0.0 <= score <= 1.0

    def test_none_snapshot_returns_original_confidence(self):
        signal = _make_signal(confidence=0.65)
        score = score_confluence(signal, None)
        assert score == 0.65

    def test_partial_indicators(self):
        """Score works with some indicators missing."""
        signal = _make_signal(Direction.LONG, price=4990.0)
        snapshot = _make_snapshot(
            rsi_14=30.0,
            vwap=None,
            ema_9=None,
            ema_21=None,
            keltner_upper=None,
            keltner_lower=None,
        )
        score = score_confluence(signal, snapshot)
        assert 0.0 <= score <= 1.0


class TestTimeOfDayAdjustment:
    def test_prime_hours_no_penalty(self):
        assert adjust_confidence_for_time_of_day(0.8, 9) == 0.8
        assert adjust_confidence_for_time_of_day(0.8, 10) == 0.8
        assert adjust_confidence_for_time_of_day(0.8, 14) == 0.8

    def test_overnight_penalty(self):
        result = adjust_confidence_for_time_of_day(0.8, 3)
        assert result < 0.8
        assert result == pytest.approx(0.6)

    def test_slight_penalty_hours(self):
        result = adjust_confidence_for_time_of_day(0.8, 7)
        assert result == pytest.approx(0.72)


class TestVolatilityAdjustment:
    def test_normal_volatility_no_change(self):
        assert adjust_confidence_for_volatility(0.8, 3.0, 3.0) == 0.8
        assert adjust_confidence_for_volatility(0.8, 4.0, 3.0) == 0.8  # 1.33 ratio

    def test_high_volatility_penalty(self):
        result = adjust_confidence_for_volatility(0.8, 7.0, 3.0)  # 2.33 ratio
        assert result < 0.8
        assert result == pytest.approx(0.48)

    def test_low_volatility_penalty(self):
        result = adjust_confidence_for_volatility(0.8, 1.0, 3.0)  # 0.33 ratio
        assert result < 0.8
        assert result == pytest.approx(0.56)

    def test_zero_avg_atr_returns_original(self):
        assert adjust_confidence_for_volatility(0.8, 3.0, 0.0) == 0.8
