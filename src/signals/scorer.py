"""Signal confidence scoring based on indicator confluence."""

from __future__ import annotations

from src.core.models import Direction, IndicatorSnapshot, Signal


def score_confluence(signal: Signal, snapshot: IndicatorSnapshot) -> float:
    """Score a signal's confluence with market context. Returns 0.0-1.0.

    Checks alignment of multiple indicators with the signal direction.
    """
    if snapshot is None:
        return signal.confidence

    score = 0.0
    factors = 0

    # BB position (is price near the expected band?)
    if snapshot.bb_lower is not None and snapshot.bb_upper is not None:
        factors += 1
        if signal.direction == Direction.LONG and signal.entry_price <= snapshot.bb_lower:
            score += 1.0
        elif signal.direction == Direction.SHORT and signal.entry_price >= snapshot.bb_upper:
            score += 1.0
        elif snapshot.bb_middle is not None:
            # Partial credit for being on the correct side of middle
            if signal.direction == Direction.LONG and signal.entry_price < snapshot.bb_middle:
                score += 0.5
            elif signal.direction == Direction.SHORT and signal.entry_price > snapshot.bb_middle:
                score += 0.5

    # RSI alignment
    if snapshot.rsi_14 is not None:
        factors += 1
        if signal.direction == Direction.LONG and snapshot.rsi_14 <= 35:
            score += 1.0
        elif signal.direction == Direction.SHORT and snapshot.rsi_14 >= 65:
            score += 1.0
        elif signal.direction == Direction.LONG and snapshot.rsi_14 <= 45:
            score += 0.5
        elif signal.direction == Direction.SHORT and snapshot.rsi_14 >= 55:
            score += 0.5

    # VWAP alignment
    if snapshot.vwap is not None:
        factors += 1
        if signal.direction == Direction.LONG and signal.entry_price < snapshot.vwap:
            score += 1.0
        elif signal.direction == Direction.SHORT and signal.entry_price > snapshot.vwap:
            score += 1.0

    # EMA alignment (short EMA vs long EMA)
    if snapshot.ema_9 is not None and snapshot.ema_21 is not None:
        factors += 1
        if signal.direction == Direction.LONG and snapshot.ema_9 < snapshot.ema_21:
            score += 1.0  # Oversold confirmation
        elif signal.direction == Direction.SHORT and snapshot.ema_9 > snapshot.ema_21:
            score += 1.0  # Overbought confirmation

    # Keltner alignment
    if snapshot.keltner_lower is not None and snapshot.keltner_upper is not None:
        factors += 1
        if signal.direction == Direction.LONG and signal.entry_price > snapshot.keltner_lower:
            score += 1.0  # Inside channel (not extreme trend)
        elif signal.direction == Direction.SHORT and signal.entry_price < snapshot.keltner_upper:
            score += 1.0

    if factors == 0:
        return signal.confidence

    return min(score / factors, 1.0)


def adjust_confidence_for_time_of_day(confidence: float, hour: int) -> float:
    """Reduce confidence during low-liquidity periods.

    Best hours (CT): 8:30-11:30 (RTH open), 13:00-15:00 (afternoon session).
    Penalty for overnight and early morning.
    """
    if 8 <= hour <= 11:
        return confidence  # Prime hours, no penalty
    if 13 <= hour <= 15:
        return confidence  # Afternoon session
    if 6 <= hour <= 7 or 12 <= hour <= 12:
        return confidence * 0.9  # Slight penalty
    # Overnight / early morning / late afternoon
    return confidence * 0.75


def adjust_confidence_for_volatility(
    confidence: float, atr: float, avg_atr: float
) -> float:
    """Adjust confidence based on whether current volatility is normal.

    Penalize if ATR is much higher (news) or much lower (chop) than average.
    """
    if avg_atr <= 0:
        return confidence

    ratio = atr / avg_atr

    if 0.7 <= ratio <= 1.5:
        return confidence  # Normal volatility
    if ratio > 2.0:
        return confidence * 0.6  # Very high vol (news event)
    if ratio > 1.5:
        return confidence * 0.8  # Elevated vol
    if ratio < 0.5:
        return confidence * 0.7  # Very low vol (chop)
    # 0.5 <= ratio < 0.7
    return confidence * 0.85
