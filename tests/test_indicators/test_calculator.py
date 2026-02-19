"""Tests for technical indicator calculations."""

import math
from datetime import datetime, timedelta

import pytest

from src.core.models import Bar
from src.indicators.calculator import IndicatorCalculator


def _make_bars(prices: list[float], base_volume: int = 1000) -> list[Bar]:
    """Create bars from a list of close prices (open=high=low=close for simplicity)."""
    base_time = datetime(2025, 1, 1, 10, 0)
    bars = []
    for i, price in enumerate(prices):
        bars.append(Bar(
            timestamp=base_time + timedelta(minutes=i),
            symbol="MES",
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=base_volume,
        ))
    return bars


def _make_ohlc_bars(data: list[tuple], base_volume: int = 1000) -> list[Bar]:
    """Create bars from (open, high, low, close) tuples."""
    bars = []
    for i, (o, h, l, c) in enumerate(data):
        bars.append(Bar(
            timestamp=datetime(2025, 1, 1, 10, i),
            symbol="MES",
            open=o,
            high=h,
            low=l,
            close=c,
            volume=base_volume,
        ))
    return bars


class TestIndicatorCalculator:
    @pytest.fixture
    def calc(self):
        return IndicatorCalculator(
            bb_period=20, bb_std=2.0,
            kc_period=20, kc_atr_multiple=1.5,
            rsi_period=14, atr_period=14,
            ema_fast=9, ema_slow=21,
        )

    def test_returns_none_with_insufficient_data(self, calc):
        """Need at least 21 bars (longest period + 1) to produce a snapshot."""
        bars = _make_bars([5000.0] * 10)
        result = None
        for bar in bars:
            result = calc.update(bar)
        assert result is None

    def test_produces_snapshot_with_enough_data(self, calc):
        """With enough bars, all indicators should be computed."""
        # Generate 50 bars with slight variation
        prices = [5000.0 + (i % 5) * 0.25 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert result.symbol == "MES"
        assert result.vwap is not None
        assert result.bb_upper is not None
        assert result.bb_middle is not None
        assert result.bb_lower is not None
        assert result.rsi_14 is not None
        assert result.atr_14 is not None
        assert result.ema_9 is not None
        assert result.ema_21 is not None

    def test_bbands_ordering(self, calc):
        """Upper > Middle > Lower for Bollinger Bands."""
        prices = [5000.0 + (i % 10) * 0.5 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert result.bb_upper > result.bb_middle > result.bb_lower

    def test_keltner_ordering(self, calc):
        """Upper > Middle > Lower for Keltner Channels."""
        prices = [5000.0 + (i % 10) * 0.5 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        if result.keltner_upper is not None:
            assert result.keltner_upper > result.keltner_middle > result.keltner_lower

    def test_rsi_in_range(self, calc):
        """RSI should be between 0 and 100."""
        prices = [5000.0 + i * 0.25 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert 0 <= result.rsi_14 <= 100

    def test_rsi_high_on_uptrend(self, calc):
        """RSI should be high (>50) on a strong uptrend."""
        prices = [5000.0 + i * 2.0 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert result.rsi_14 > 50

    def test_rsi_low_on_downtrend(self, calc):
        """RSI should be low (<50) on a strong downtrend."""
        prices = [5100.0 - i * 2.0 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert result.rsi_14 < 50

    def test_atr_positive(self, calc):
        """ATR should always be positive."""
        prices = [5000.0 + (i % 10) * 0.5 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert result.atr_14 > 0

    def test_ema_fast_responds_faster(self, calc):
        """EMA-9 should respond faster to recent price changes than EMA-21."""
        # Start flat, then sharp move up
        prices = [5000.0] * 30 + [5010.0] * 20
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        # After the jump, EMA-9 should be closer to current price than EMA-21
        assert result.ema_9 > result.ema_21

    def test_vwap_basic(self, calc):
        """VWAP should be between high and low of the session."""
        prices = [5000.0 + i * 0.25 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert result.vwap is not None
        # VWAP should be somewhere in the price range
        min_price = min(b.low for b in bars)
        max_price = max(b.high for b in bars)
        assert min_price <= result.vwap <= max_price

    def test_vwap_resets_on_new_day(self, calc):
        """VWAP should reset when a new session date starts."""
        # Day 1 bars
        day1_bars = []
        for i in range(30):
            day1_bars.append(Bar(
                timestamp=datetime(2025, 1, 1, 10, i),
                symbol="MES", open=5000, high=5000.5,
                low=4999.5, close=5000, volume=1000,
            ))

        # Day 2 bars at a different price
        day2_bars = []
        for i in range(30):
            day2_bars.append(Bar(
                timestamp=datetime(2025, 1, 2, 10, i),
                symbol="MES", open=5100, high=5100.5,
                low=5099.5, close=5100, volume=1000,
            ))

        for bar in day1_bars:
            calc.update(bar)

        result = None
        for bar in day2_bars:
            result = calc.update(bar)

        assert result is not None
        # VWAP should be near 5100, not averaging with day1's 5000
        assert result.vwap > 5050

    def test_buffer_doesnt_grow_unbounded(self, calc):
        """Internal bar buffer should be capped."""
        prices = [5000.0 + (i % 10) * 0.25 for i in range(200)]
        bars = _make_bars(prices)

        for bar in bars:
            calc.update(bar)

        assert len(calc._bars) <= calc._max_bars

    def test_snapshot_timestamp_matches_bar(self, calc):
        """Snapshot timestamp should match the bar that triggered it."""
        prices = [5000.0 + (i % 5) * 0.25 for i in range(50)]
        bars = _make_bars(prices)

        result = None
        for bar in bars:
            result = calc.update(bar)

        assert result is not None
        assert result.timestamp == bars[-1].timestamp
