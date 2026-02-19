"""Mean reversion strategy: Bollinger Band touch + RSI confirmation.

Entry conditions:
  LONG:  close <= bb_lower AND rsi_14 <= rsi_oversold AND close > keltner_lower
  SHORT: close >= bb_upper AND rsi_14 >= rsi_overbought AND close < keltner_upper

The Keltner Channel filter prevents entries during extreme trends.
"""

from __future__ import annotations

from src.core.models import Bar, Direction, IndicatorSnapshot, Signal
from src.risk.stop_loss import calculate_initial_stop, calculate_take_profit
from src.strategies.base import BaseStrategy, StrategyConfig

DEFAULT_PARAMS = {
    "bb_touch_threshold": 0.0,
    "rsi_oversold": 35.0,
    "rsi_overbought": 65.0,
    "rsi_extreme_oversold": 25.0,
    "rsi_extreme_overbought": 75.0,
    "atr_stop_multiple": 1.5,
    "risk_reward_target": 2.0,
    "require_keltner_filter": True,
    "require_vwap_alignment": False,
    "min_atr": 0.5,
}


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band mean reversion with RSI confirmation."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        if config is None:
            config = StrategyConfig(name="mean_reversion", params=DEFAULT_PARAMS.copy())
        else:
            merged = {**DEFAULT_PARAMS, **config.params}
            config.params = merged
        super().__init__(config)
        self.validate_params()

    @property
    def params(self) -> dict:
        return self.config.params

    def generate_signal(self, bar: Bar, snapshot: IndicatorSnapshot) -> Signal | None:
        """Check BB+RSI conditions and generate signal if met."""
        if not self._indicators_ready(snapshot):
            return None

        if snapshot.atr_14 < self.params["min_atr"]:
            return None

        direction = self._check_entry_conditions(bar, snapshot)
        if direction is None:
            return None

        confidence = self._calculate_confidence(bar, snapshot, direction)
        if confidence < self.config.min_confidence:
            return None

        stop_price = calculate_initial_stop(
            entry_price=bar.close,
            direction=direction,
            atr=snapshot.atr_14,
            atr_multiple=self.params["atr_stop_multiple"],
        )
        take_profit = calculate_take_profit(
            entry_price=bar.close,
            stop_price=stop_price,
            direction=direction,
            risk_reward_ratio=self.params["risk_reward_target"],
        )

        reason = self._build_reason(bar, snapshot, direction)

        return Signal(
            strategy=self.name,
            symbol=bar.symbol,
            direction=direction,
            confidence=confidence,
            entry_price=bar.close,
            stop_loss=stop_price,
            take_profit=take_profit,
            reason=reason,
            market_context=snapshot,
        )

    def _check_entry_conditions(
        self, bar: Bar, snap: IndicatorSnapshot
    ) -> Direction | None:
        threshold = self.params["bb_touch_threshold"]

        # LONG: price at or below lower BB + RSI oversold
        long_bb = bar.close <= snap.bb_lower + threshold
        long_rsi = snap.rsi_14 <= self.params["rsi_oversold"]
        long_keltner = (
            not self.params["require_keltner_filter"]
            or snap.keltner_lower is None
            or bar.close > snap.keltner_lower
        )

        if long_bb and long_rsi and long_keltner:
            if self.params["require_vwap_alignment"] and snap.vwap is not None:
                if bar.close > snap.vwap:
                    return None
            return Direction.LONG

        # SHORT: price at or above upper BB + RSI overbought
        short_bb = bar.close >= snap.bb_upper - threshold
        short_rsi = snap.rsi_14 >= self.params["rsi_overbought"]
        short_keltner = (
            not self.params["require_keltner_filter"]
            or snap.keltner_upper is None
            or bar.close < snap.keltner_upper
        )

        if short_bb and short_rsi and short_keltner:
            if self.params["require_vwap_alignment"] and snap.vwap is not None:
                if bar.close < snap.vwap:
                    return None
            return Direction.SHORT

        return None

    def _calculate_confidence(
        self, bar: Bar, snap: IndicatorSnapshot, direction: Direction
    ) -> float:
        score = 0.5

        # RSI extremity bonus (up to +0.15)
        if direction == Direction.LONG:
            if snap.rsi_14 <= self.params["rsi_extreme_oversold"]:
                score += 0.15
            elif snap.rsi_14 <= self.params["rsi_oversold"] - 5:
                score += 0.08
        else:
            if snap.rsi_14 >= self.params["rsi_extreme_overbought"]:
                score += 0.15
            elif snap.rsi_14 >= self.params["rsi_overbought"] + 5:
                score += 0.08

        # VWAP alignment bonus (+0.10)
        if snap.vwap is not None:
            if direction == Direction.LONG and bar.close < snap.vwap:
                score += 0.10
            elif direction == Direction.SHORT and bar.close > snap.vwap:
                score += 0.10

        # EMA alignment bonus (+0.10)
        if snap.ema_9 is not None and snap.ema_21 is not None:
            if direction == Direction.LONG and snap.ema_9 < snap.ema_21:
                score += 0.10
            elif direction == Direction.SHORT and snap.ema_9 > snap.ema_21:
                score += 0.10

        # BB penetration depth bonus (+0.05-0.10)
        if direction == Direction.LONG and snap.bb_lower is not None:
            penetration = (snap.bb_lower - bar.close) / snap.atr_14 if snap.atr_14 else 0
            if penetration > 0.5:
                score += 0.10
            elif penetration > 0.2:
                score += 0.05
        elif direction == Direction.SHORT and snap.bb_upper is not None:
            penetration = (bar.close - snap.bb_upper) / snap.atr_14 if snap.atr_14 else 0
            if penetration > 0.5:
                score += 0.10
            elif penetration > 0.2:
                score += 0.05

        return min(score, 1.0)

    def _indicators_ready(self, snap: IndicatorSnapshot) -> bool:
        return all([
            snap.bb_upper is not None,
            snap.bb_lower is not None,
            snap.bb_middle is not None,
            snap.rsi_14 is not None,
            snap.atr_14 is not None,
        ])

    def _build_reason(
        self, bar: Bar, snap: IndicatorSnapshot, direction: Direction
    ) -> str:
        parts = []
        if direction == Direction.LONG:
            parts.append(f"BB lower touch (close={bar.close:.2f}, bb_lower={snap.bb_lower:.2f})")
            parts.append(f"RSI oversold ({snap.rsi_14:.1f})")
        else:
            parts.append(f"BB upper touch (close={bar.close:.2f}, bb_upper={snap.bb_upper:.2f})")
            parts.append(f"RSI overbought ({snap.rsi_14:.1f})")
        if snap.vwap:
            parts.append(f"VWAP={snap.vwap:.2f}")
        return " | ".join(parts)

    def validate_params(self) -> bool:
        p = self.params
        assert 0 < p["rsi_oversold"] < 50, "rsi_oversold must be 0-50"
        assert 50 < p["rsi_overbought"] < 100, "rsi_overbought must be 50-100"
        assert p["atr_stop_multiple"] > 0, "atr_stop_multiple must be positive"
        assert p["risk_reward_target"] > 0, "risk_reward_target must be positive"
        return True
