"""Stop-loss calculation and trailing stop logic.

ATR-based initial stop placement with trailing activation after 1R profit.
"""

from __future__ import annotations

from src.config import RISK_DEFAULTS, MES_SPEC
from src.core.models import Direction


def calculate_initial_stop(
    entry_price: float,
    direction: Direction,
    atr: float,
    atr_multiple: float = 1.5,
    tick_size: float = MES_SPEC["tick_size"],
) -> float:
    """Calculate initial stop-loss price based on ATR.

    Places stop at atr_multiple * ATR from entry, rounded to tick size.
    """
    if atr <= 0:
        raise ValueError("ATR must be positive")

    stop_distance = atr * atr_multiple

    if direction == Direction.LONG:
        raw_stop = entry_price - stop_distance
    else:
        raw_stop = entry_price + stop_distance

    # Round to nearest tick
    return _round_to_tick(raw_stop, tick_size)


def calculate_stop_distance_ticks(
    entry_price: float,
    stop_price: float,
    direction: Direction,
    tick_size: float = MES_SPEC["tick_size"],
) -> float:
    """Calculate stop distance in ticks."""
    if direction == Direction.LONG:
        distance = entry_price - stop_price
    else:
        distance = stop_price - entry_price

    if distance <= 0:
        return 0.0

    return distance / tick_size


def calculate_take_profit(
    entry_price: float,
    stop_price: float,
    direction: Direction,
    risk_reward_ratio: float = RISK_DEFAULTS["min_risk_reward_ratio"],
    tick_size: float = MES_SPEC["tick_size"],
) -> float:
    """Calculate take-profit price for a given risk:reward ratio."""
    if direction == Direction.LONG:
        risk = entry_price - stop_price
        raw_target = entry_price + (risk * risk_reward_ratio)
    else:
        risk = stop_price - entry_price
        raw_target = entry_price - (risk * risk_reward_ratio)

    return _round_to_tick(raw_target, tick_size)


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: Direction,
) -> float:
    """Calculate the risk:reward ratio for a trade setup."""
    if direction == Direction.LONG:
        risk = entry_price - stop_price
        reward = target_price - entry_price
    else:
        risk = stop_price - entry_price
        reward = entry_price - target_price

    if risk <= 0:
        return 0.0
    return reward / risk


def update_trailing_stop(
    entry_price: float,
    current_price: float,
    current_stop: float,
    direction: Direction,
    atr: float,
    trailing_atr_multiple: float = 1.5,
    activation_r_multiple: float = 1.0,
    tick_size: float = MES_SPEC["tick_size"],
) -> tuple[float, bool]:
    """Update trailing stop if conditions are met.

    Trailing activates after price moves 1R in favor (activation_r_multiple).
    Once activated, trails at trailing_atr_multiple * ATR from the extreme.

    Returns (new_stop_price, trailing_activated).
    """
    if atr <= 0:
        return current_stop, False

    # Calculate 1R distance (from entry to initial stop)
    if direction == Direction.LONG:
        initial_risk = entry_price - current_stop
        current_profit = current_price - entry_price
    else:
        initial_risk = current_stop - entry_price
        current_profit = entry_price - current_price

    if initial_risk <= 0:
        return current_stop, False

    # Check if trailing should activate
    activation_threshold = initial_risk * activation_r_multiple
    if current_profit < activation_threshold:
        return current_stop, False

    # Trailing is active — calculate new trailing stop
    trail_distance = atr * trailing_atr_multiple

    if direction == Direction.LONG:
        new_stop = current_price - trail_distance
        new_stop = _round_to_tick(new_stop, tick_size)
        # Only move stop UP (never down)
        if new_stop > current_stop:
            return new_stop, True
    else:
        new_stop = current_price + trail_distance
        new_stop = _round_to_tick(new_stop, tick_size)
        # Only move stop DOWN (never up)
        if new_stop < current_stop:
            return new_stop, True

    return current_stop, True


def validate_stop_placement(
    entry_price: float,
    stop_price: float,
    direction: Direction,
    atr: float,
    max_atr_multiple: float = RISK_DEFAULTS["max_stop_distance_atr"],
    tick_size: float = MES_SPEC["tick_size"],
) -> tuple[bool, str]:
    """Validate that a stop-loss placement is acceptable.

    Returns (valid, reason).
    """
    if direction == Direction.LONG:
        if stop_price >= entry_price:
            return False, "Long stop must be below entry price"
        distance = entry_price - stop_price
    else:
        if stop_price <= entry_price:
            return False, "Short stop must be above entry price"
        distance = stop_price - entry_price

    if distance <= 0:
        return False, "Stop distance must be positive"

    if atr > 0:
        max_distance = atr * max_atr_multiple
        if distance > max_distance:
            return False, (
                f"Stop distance ({distance:.2f}) exceeds "
                f"{max_atr_multiple}x ATR ({max_distance:.2f})"
            )

    # Check it aligns with tick size
    ticks = distance / tick_size
    if abs(ticks - round(ticks)) > 0.001:
        return False, f"Stop distance not aligned to tick size ({tick_size})"

    return True, ""


def _round_to_tick(price: float, tick_size: float) -> float:
    """Round a price to the nearest valid tick."""
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)
