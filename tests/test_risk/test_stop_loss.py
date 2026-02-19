"""Tests for stop-loss calculation and trailing stop logic."""

import pytest

from src.core.models import Direction
from src.risk.stop_loss import (
    _round_to_tick,
    calculate_initial_stop,
    calculate_risk_reward_ratio,
    calculate_stop_distance_ticks,
    calculate_take_profit,
    update_trailing_stop,
    validate_stop_placement,
)


class TestCalculateInitialStop:
    def test_long_stop_below_entry(self):
        stop = calculate_initial_stop(5000.00, Direction.LONG, atr=3.0, atr_multiple=1.5)
        # 5000 - (3.0 * 1.5) = 4995.5 -> rounded to 4995.50
        assert stop == 4995.50

    def test_short_stop_above_entry(self):
        stop = calculate_initial_stop(5000.00, Direction.SHORT, atr=3.0, atr_multiple=1.5)
        # 5000 + 4.5 = 5004.5 -> 5004.50
        assert stop == 5004.50

    def test_rounds_to_tick_size(self):
        # ATR=2.7, multiple=1.5 -> distance=4.05
        # 5000 - 4.05 = 4995.95, rounded to nearest 0.25 = 4996.00
        stop = calculate_initial_stop(5000.00, Direction.LONG, atr=2.7, atr_multiple=1.5)
        assert stop % 0.25 == pytest.approx(0, abs=0.001)

    def test_zero_atr_raises(self):
        with pytest.raises(ValueError, match="ATR must be positive"):
            calculate_initial_stop(5000, Direction.LONG, atr=0)

    def test_negative_atr_raises(self):
        with pytest.raises(ValueError, match="ATR must be positive"):
            calculate_initial_stop(5000, Direction.LONG, atr=-1.5)


class TestCalculateStopDistanceTicks:
    def test_long_distance(self):
        ticks = calculate_stop_distance_ticks(5000.00, 4996.00, Direction.LONG)
        assert ticks == 16.0  # 4 points / 0.25 tick = 16 ticks

    def test_short_distance(self):
        ticks = calculate_stop_distance_ticks(5000.00, 5004.00, Direction.SHORT)
        assert ticks == 16.0

    def test_invalid_long_stop_above_entry(self):
        ticks = calculate_stop_distance_ticks(5000.00, 5002.00, Direction.LONG)
        assert ticks == 0.0

    def test_invalid_short_stop_below_entry(self):
        ticks = calculate_stop_distance_ticks(5000.00, 4998.00, Direction.SHORT)
        assert ticks == 0.0

    def test_same_price(self):
        ticks = calculate_stop_distance_ticks(5000.00, 5000.00, Direction.LONG)
        assert ticks == 0.0


class TestCalculateTakeProfit:
    def test_long_2r(self):
        tp = calculate_take_profit(5000.00, 4996.00, Direction.LONG, 2.0)
        # Risk = 4 points, reward = 8 points -> 5008.00
        assert tp == 5008.00

    def test_short_2r(self):
        tp = calculate_take_profit(5000.00, 5004.00, Direction.SHORT, 2.0)
        # Risk = 4 points, reward = 8 points -> 4992.00
        assert tp == 4992.00

    def test_long_3r(self):
        tp = calculate_take_profit(5000.00, 4996.00, Direction.LONG, 3.0)
        assert tp == 5012.00

    def test_rounds_to_tick(self):
        tp = calculate_take_profit(5000.00, 4997.00, Direction.LONG, 2.0)
        assert tp % 0.25 == pytest.approx(0, abs=0.001)


class TestCalculateRiskRewardRatio:
    def test_2_to_1_long(self):
        rr = calculate_risk_reward_ratio(5000, 4996, 5008, Direction.LONG)
        assert rr == pytest.approx(2.0)

    def test_2_to_1_short(self):
        rr = calculate_risk_reward_ratio(5000, 5004, 4992, Direction.SHORT)
        assert rr == pytest.approx(2.0)

    def test_less_than_1_to_1(self):
        rr = calculate_risk_reward_ratio(5000, 4996, 5002, Direction.LONG)
        assert rr == pytest.approx(0.5)

    def test_zero_risk(self):
        rr = calculate_risk_reward_ratio(5000, 5000, 5008, Direction.LONG)
        assert rr == 0.0

    def test_negative_reward(self):
        rr = calculate_risk_reward_ratio(5000, 4996, 4998, Direction.LONG)
        assert rr < 0  # Loss scenario


class TestUpdateTrailingStop:
    def test_no_activation_before_1r(self):
        """Trailing shouldn't activate until 1R profit."""
        new_stop, activated = update_trailing_stop(
            entry_price=5000,
            current_price=5002,  # Only 2 points profit, risk is 4 points
            current_stop=4996,
            direction=Direction.LONG,
            atr=3.0,
            trailing_atr_multiple=1.5,
        )
        assert new_stop == 4996  # Unchanged
        assert activated is False

    def test_activation_at_1r(self):
        """Trailing activates at 1R profit."""
        new_stop, activated = update_trailing_stop(
            entry_price=5000,
            current_price=5005,  # 5 pts profit, risk is 4 pts (> 1R)
            current_stop=4996,
            direction=Direction.LONG,
            atr=3.0,
            trailing_atr_multiple=1.5,
        )
        # Trail: 5005 - 4.5 = 5000.50 -> rounded to 5000.50
        assert activated is True
        assert new_stop > 4996  # Moved up

    def test_trailing_only_moves_up_for_long(self):
        """Long trailing stop never moves down."""
        new_stop, _ = update_trailing_stop(
            entry_price=5000,
            current_price=5005,
            current_stop=5002,  # Already above where trail would place it
            direction=Direction.LONG,
            atr=3.0,
            trailing_atr_multiple=1.5,
        )
        # 5005 - 4.5 = 5000.5, but current_stop is 5002 which is higher
        assert new_stop == 5002  # Unchanged

    def test_short_trailing_only_moves_down(self):
        """Short trailing stop never moves up."""
        new_stop, activated = update_trailing_stop(
            entry_price=5000,
            current_price=4994,  # 6 pts profit, risk=4 pts
            current_stop=5004,
            direction=Direction.SHORT,
            atr=3.0,
            trailing_atr_multiple=1.5,
        )
        # Trail: 4994 + 4.5 = 4998.5
        assert activated is True
        assert new_stop < 5004

    def test_zero_atr(self):
        new_stop, activated = update_trailing_stop(
            entry_price=5000,
            current_price=5010,
            current_stop=4996,
            direction=Direction.LONG,
            atr=0,
        )
        assert new_stop == 4996
        assert activated is False

    def test_zero_initial_risk(self):
        """Stop at entry means zero risk, no trailing."""
        new_stop, activated = update_trailing_stop(
            entry_price=5000,
            current_price=5010,
            current_stop=5000,  # Stop = entry
            direction=Direction.LONG,
            atr=3.0,
        )
        assert new_stop == 5000
        assert activated is False


class TestValidateStopPlacement:
    def test_valid_long_stop(self):
        valid, _ = validate_stop_placement(5000, 4996, Direction.LONG, atr=3.0)
        assert valid is True

    def test_valid_short_stop(self):
        valid, _ = validate_stop_placement(5000, 5004, Direction.SHORT, atr=3.0)
        assert valid is True

    def test_long_stop_above_entry(self):
        valid, reason = validate_stop_placement(5000, 5002, Direction.LONG, atr=3.0)
        assert valid is False
        assert "below entry" in reason

    def test_short_stop_below_entry(self):
        valid, reason = validate_stop_placement(5000, 4998, Direction.SHORT, atr=3.0)
        assert valid is False
        assert "above entry" in reason

    def test_stop_at_entry(self):
        valid, reason = validate_stop_placement(5000, 5000, Direction.LONG, atr=3.0)
        assert valid is False

    def test_exceeds_atr_limit(self):
        # 2x ATR = 6 points max. Distance = 8 points.
        valid, reason = validate_stop_placement(
            5000, 4992, Direction.LONG, atr=3.0, max_atr_multiple=2.0
        )
        assert valid is False
        assert "exceeds" in reason

    def test_exactly_at_atr_limit(self):
        # 2x ATR = 6 points. Distance = 6 points.
        valid, _ = validate_stop_placement(
            5000, 4994, Direction.LONG, atr=3.0, max_atr_multiple=2.0
        )
        assert valid is True


class TestRoundToTick:
    def test_exact_tick(self):
        assert _round_to_tick(5000.25, 0.25) == 5000.25

    def test_rounds_up(self):
        assert _round_to_tick(5000.13, 0.25) == 5000.25

    def test_rounds_down(self):
        assert _round_to_tick(5000.10, 0.25) == 5000.00

    def test_midpoint_rounds(self):
        # 5000.125 is exactly between 5000.00 and 5000.25
        result = _round_to_tick(5000.125, 0.25)
        assert result in (5000.00, 5000.25)  # Either is acceptable

    def test_zero_tick_size(self):
        assert _round_to_tick(5000.13, 0) == 5000.13  # No rounding
