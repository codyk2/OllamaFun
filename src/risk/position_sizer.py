"""Fixed-fractional position sizing.

Formula: contracts = (account_equity * max_risk_pct) / (stop_distance_ticks * tick_value)

Always returns 0 (no trade) if risk limits would be exceeded.
"""

from __future__ import annotations

import math

from src.config import RISK_DEFAULTS, MES_SPEC


def calculate_position_size(
    account_equity: float,
    stop_distance_ticks: float,
    tick_value: float = MES_SPEC["tick_value"],
    max_risk_pct: float = RISK_DEFAULTS["max_risk_per_trade"],
    max_position_size: int = RISK_DEFAULTS["max_position_size"],
) -> int:
    """Calculate the number of contracts to trade.

    Returns 0 if the trade cannot be sized safely.
    """
    if account_equity <= 0:
        return 0
    if stop_distance_ticks <= 0:
        return 0
    if tick_value <= 0:
        return 0
    if max_risk_pct <= 0:
        return 0

    max_risk_dollars = account_equity * max_risk_pct
    risk_per_contract = stop_distance_ticks * tick_value

    if risk_per_contract <= 0:
        return 0

    raw_size = max_risk_dollars / risk_per_contract
    contracts = math.floor(raw_size)

    # Enforce hard cap
    contracts = min(contracts, max_position_size)

    # Must be at least 0
    return max(contracts, 0)


def validate_stop_distance(
    stop_distance_ticks: float,
    atr_ticks: float,
    max_atr_multiple: float = RISK_DEFAULTS["max_stop_distance_atr"],
) -> bool:
    """Check if stop distance is within allowed ATR multiple."""
    if atr_ticks <= 0:
        return False
    if stop_distance_ticks <= 0:
        return False
    return stop_distance_ticks <= (atr_ticks * max_atr_multiple)


def calculate_risk_dollars(
    stop_distance_ticks: float,
    quantity: int,
    tick_value: float = MES_SPEC["tick_value"],
    commission_per_side: float = MES_SPEC["commission_per_side"],
) -> float:
    """Calculate total risk in dollars for a given position."""
    if quantity <= 0 or stop_distance_ticks <= 0:
        return 0.0
    risk_from_stop = stop_distance_ticks * tick_value * quantity
    commission_total = commission_per_side * 2 * quantity  # round-trip
    return risk_from_stop + commission_total


def risk_as_percent_of_equity(
    risk_dollars: float,
    account_equity: float,
) -> float:
    """Express risk as a percentage of account equity."""
    if account_equity <= 0:
        return float("inf")
    return risk_dollars / account_equity
