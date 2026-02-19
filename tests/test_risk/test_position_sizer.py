"""Tests for position sizing — 100% coverage required."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.risk.position_sizer import (
    calculate_position_size,
    calculate_risk_dollars,
    risk_as_percent_of_equity,
    validate_stop_distance,
)


class TestCalculatePositionSize:
    """Tests for the fixed-fractional position sizer."""

    def test_basic_calculation(self):
        """Standard case: $10k equity, 8 tick stop, 1.25 tick value."""
        # max risk = 10000 * 0.015 = $150
        # risk per contract = 8 * 1.25 = $10
        # contracts = floor(150 / 10) = 15, but capped at max_position_size=2
        size = calculate_position_size(
            account_equity=10000,
            stop_distance_ticks=8,
            tick_value=1.25,
            max_risk_pct=0.015,
            max_position_size=2,
        )
        assert size == 2  # capped

    def test_small_account_gets_one_contract(self):
        """Very small account should get 1 contract if affordable."""
        size = calculate_position_size(
            account_equity=1000,
            stop_distance_ticks=8,
            tick_value=1.25,
            max_risk_pct=0.015,
            max_position_size=5,
        )
        # max risk = 1000 * 0.015 = $15
        # risk per contract = 8 * 1.25 = $10
        # floor(15/10) = 1
        assert size == 1

    def test_too_small_account_gets_zero(self):
        """Account too small for even 1 contract."""
        size = calculate_position_size(
            account_equity=100,
            stop_distance_ticks=20,
            tick_value=1.25,
            max_risk_pct=0.015,
            max_position_size=5,
        )
        # max risk = 100 * 0.015 = $1.50
        # risk per contract = 20 * 1.25 = $25
        # floor(1.50/25) = 0
        assert size == 0

    def test_respects_max_position_size(self):
        """Never exceeds max position size even with large account."""
        size = calculate_position_size(
            account_equity=1_000_000,
            stop_distance_ticks=4,
            tick_value=1.25,
            max_risk_pct=0.015,
            max_position_size=2,
        )
        assert size == 2

    def test_zero_equity_returns_zero(self):
        size = calculate_position_size(account_equity=0, stop_distance_ticks=8)
        assert size == 0

    def test_negative_equity_returns_zero(self):
        size = calculate_position_size(account_equity=-5000, stop_distance_ticks=8)
        assert size == 0

    def test_zero_stop_distance_returns_zero(self):
        size = calculate_position_size(account_equity=10000, stop_distance_ticks=0)
        assert size == 0

    def test_negative_stop_distance_returns_zero(self):
        size = calculate_position_size(account_equity=10000, stop_distance_ticks=-5)
        assert size == 0

    def test_zero_tick_value_returns_zero(self):
        size = calculate_position_size(
            account_equity=10000, stop_distance_ticks=8, tick_value=0
        )
        assert size == 0

    def test_negative_tick_value_returns_zero(self):
        size = calculate_position_size(
            account_equity=10000, stop_distance_ticks=8, tick_value=-1.25
        )
        assert size == 0

    def test_zero_risk_pct_returns_zero(self):
        size = calculate_position_size(
            account_equity=10000, stop_distance_ticks=8, max_risk_pct=0
        )
        assert size == 0

    def test_negative_risk_pct_returns_zero(self):
        size = calculate_position_size(
            account_equity=10000, stop_distance_ticks=8, max_risk_pct=-0.01
        )
        assert size == 0

    @given(
        equity=st.floats(min_value=0.01, max_value=1_000_000, allow_nan=False),
        stop_ticks=st.floats(min_value=0.25, max_value=100, allow_nan=False),
        tick_val=st.floats(min_value=0.01, max_value=50, allow_nan=False),
        risk_pct=st.floats(min_value=0.001, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_never_exceeds_risk_limit(self, equity, stop_ticks, tick_val, risk_pct):
        """Property: position risk never exceeds max_risk_pct of equity."""
        size = calculate_position_size(
            account_equity=equity,
            stop_distance_ticks=stop_ticks,
            tick_value=tick_val,
            max_risk_pct=risk_pct,
            max_position_size=100,  # high cap to test risk math
        )
        if size > 0:
            actual_risk = size * stop_ticks * tick_val
            max_allowed = equity * risk_pct
            assert actual_risk <= max_allowed + 0.01  # small float tolerance

    @given(
        equity=st.floats(min_value=0.01, max_value=1_000_000, allow_nan=False),
        stop_ticks=st.floats(min_value=0.25, max_value=100, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_always_non_negative(self, equity, stop_ticks):
        """Property: position size is always >= 0."""
        size = calculate_position_size(
            account_equity=equity, stop_distance_ticks=stop_ticks
        )
        assert size >= 0

    @given(
        equity=st.floats(min_value=0.01, max_value=1_000_000, allow_nan=False),
        stop_ticks=st.floats(min_value=0.25, max_value=100, allow_nan=False),
        max_pos=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_never_exceeds_max_position(self, equity, stop_ticks, max_pos):
        """Property: never exceeds max_position_size."""
        size = calculate_position_size(
            account_equity=equity,
            stop_distance_ticks=stop_ticks,
            max_position_size=max_pos,
        )
        assert size <= max_pos


class TestValidateStopDistance:
    def test_valid_distance(self):
        assert validate_stop_distance(8, 6, 2.0) is True

    def test_exactly_at_limit(self):
        assert validate_stop_distance(12, 6, 2.0) is True

    def test_exceeds_limit(self):
        assert validate_stop_distance(13, 6, 2.0) is False

    def test_zero_atr_returns_false(self):
        assert validate_stop_distance(8, 0, 2.0) is False

    def test_negative_atr_returns_false(self):
        assert validate_stop_distance(8, -5, 2.0) is False

    def test_zero_stop_returns_false(self):
        assert validate_stop_distance(0, 6, 2.0) is False

    def test_negative_stop_returns_false(self):
        assert validate_stop_distance(-3, 6, 2.0) is False


class TestCalculateRiskDollars:
    def test_basic(self):
        # 8 ticks * $1.25 * 1 contract + $1.24 round-trip = $11.24
        risk = calculate_risk_dollars(8, 1, 1.25, 0.62)
        assert abs(risk - 11.24) < 0.01

    def test_multiple_contracts(self):
        risk = calculate_risk_dollars(8, 2, 1.25, 0.62)
        # 8 * 1.25 * 2 + 0.62*2*2 = 20 + 2.48 = 22.48
        assert abs(risk - 22.48) < 0.01

    def test_zero_quantity(self):
        assert calculate_risk_dollars(8, 0) == 0.0

    def test_zero_stop(self):
        assert calculate_risk_dollars(0, 1) == 0.0


class TestRiskAsPercent:
    def test_basic(self):
        pct = risk_as_percent_of_equity(150, 10000)
        assert abs(pct - 0.015) < 0.0001

    def test_zero_equity(self):
        pct = risk_as_percent_of_equity(150, 0)
        assert pct == float("inf")

    def test_negative_equity(self):
        pct = risk_as_percent_of_equity(150, -1000)
        assert pct == float("inf")
