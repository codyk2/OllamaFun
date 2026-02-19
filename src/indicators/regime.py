"""Market regime detection using ADX + Bollinger/Keltner squeeze.

Classifies market into three regimes with graduated signal scaling:
  RANGING      -> ADX < 20 or squeeze active   -> signal_scaling = 1.0
  TRANSITIONAL -> 20 <= ADX < 30, no squeeze   -> signal_scaling = 0.5
  TRENDING     -> ADX >= 30, no squeeze         -> signal_scaling = 0.0

Uses 5-minute bars for stability. In backtesting, aggregates 1-min bars
into 5-min bars internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import pandas_ta as ta

from src.core.logging import get_logger
from src.core.models import Bar

logger = get_logger("regime")

# Regime thresholds
ADX_RANGING_THRESHOLD = 20.0
ADX_TRENDING_THRESHOLD = 30.0
HYSTERESIS_BARS = 3  # Consecutive bars required before regime switch


class MarketRegime(str, Enum):
    RANGING = "RANGING"
    TRANSITIONAL = "TRANSITIONAL"
    TRENDING = "TRENDING"


# Signal scaling by regime: mean reversion works best in ranging markets
REGIME_SCALING = {
    MarketRegime.RANGING: 1.0,
    MarketRegime.TRANSITIONAL: 0.5,
    MarketRegime.TRENDING: 0.0,
}


@dataclass
class RegimeState:
    """Current market regime classification."""

    regime: MarketRegime = MarketRegime.RANGING
    adx: float | None = None
    squeeze_active: bool = False
    signal_scaling: float = 1.0


class RegimeDetector:
    """Detects market regime from 5-minute bar data.

    Aggregates 1-min bars into 5-min bars, computes ADX and BB/KC squeeze,
    classifies regime with hysteresis to prevent whipsawing.
    """

    def __init__(
        self,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_atr_multiple: float = 1.5,
    ) -> None:
        self.adx_period = adx_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_multiple = kc_atr_multiple

        self._5m_bars: list[Bar] = []
        self._1m_buffer: list[Bar] = []
        self._max_5m_bars = max(adx_period, bb_period, kc_period) + 50

        self._state = RegimeState()
        self._candidate_regime: MarketRegime | None = None
        self._candidate_count: int = 0

    @property
    def state(self) -> RegimeState:
        return self._state

    def on_5m_bar(self, bar: Bar) -> RegimeState:
        """Process a 5-minute bar and update regime classification."""
        self._5m_bars.append(bar)
        if len(self._5m_bars) > self._max_5m_bars:
            self._5m_bars = self._5m_bars[-self._max_5m_bars:]

        self._update_regime()
        return self._state

    def on_1m_bar(self, bar: Bar) -> RegimeState:
        """Aggregate 1-min bars into 5-min bars internally."""
        self._1m_buffer.append(bar)

        if len(self._1m_buffer) >= 5:
            bars_to_aggregate = self._1m_buffer[:5]
            self._1m_buffer = self._1m_buffer[5:]

            bar_5m = Bar(
                timestamp=bars_to_aggregate[-1].timestamp,
                symbol=bar.symbol,
                open=bars_to_aggregate[0].open,
                high=max(b.high for b in bars_to_aggregate),
                low=min(b.low for b in bars_to_aggregate),
                close=bars_to_aggregate[-1].close,
                volume=sum(b.volume for b in bars_to_aggregate),
            )
            return self.on_5m_bar(bar_5m)

        return self._state

    def _update_regime(self) -> None:
        """Classify regime using ADX + squeeze, with hysteresis."""
        min_required = max(self.adx_period, self.bb_period, self.kc_period) + 1
        if len(self._5m_bars) < min_required:
            return

        df = pd.DataFrame({
            "open": [b.open for b in self._5m_bars],
            "high": [b.high for b in self._5m_bars],
            "low": [b.low for b in self._5m_bars],
            "close": [b.close for b in self._5m_bars],
            "volume": [b.volume for b in self._5m_bars],
        })

        adx_val = self._compute_adx(df)
        squeeze = self._check_squeeze(df)

        self._state.adx = adx_val
        self._state.squeeze_active = squeeze

        if adx_val is None:
            return

        # Determine raw regime from current values
        if squeeze or adx_val < ADX_RANGING_THRESHOLD:
            raw_regime = MarketRegime.RANGING
        elif adx_val >= ADX_TRENDING_THRESHOLD:
            raw_regime = MarketRegime.TRENDING
        else:
            raw_regime = MarketRegime.TRANSITIONAL

        # Apply hysteresis: require HYSTERESIS_BARS consecutive bars
        if raw_regime != self._state.regime:
            if raw_regime == self._candidate_regime:
                self._candidate_count += 1
            else:
                self._candidate_regime = raw_regime
                self._candidate_count = 1

            if self._candidate_count >= HYSTERESIS_BARS:
                self._state.regime = raw_regime
                self._state.signal_scaling = REGIME_SCALING[raw_regime]
                self._candidate_regime = None
                self._candidate_count = 0
                logger.info(
                    "regime_change",
                    regime=raw_regime.value,
                    adx=f"{adx_val:.1f}",
                    squeeze=squeeze,
                )
        else:
            self._candidate_regime = None
            self._candidate_count = 0

    def _compute_adx(self, df: pd.DataFrame) -> float | None:
        """Compute ADX from 5-min bar dataframe."""
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
            if adx_df is None or adx_df.empty:
                return None
            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
            if not adx_col:
                return None
            val = adx_df[adx_col[0]].iloc[-1]
            return float(val) if pd.notna(val) else None
        except Exception as e:
            logger.error("adx_calc_error", error=str(e))
            return None

    def _check_squeeze(self, df: pd.DataFrame) -> bool:
        """Check if BB is inside KC (squeeze = ranging/consolidation)."""
        try:
            bbands = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
            kc = ta.kc(
                df["high"], df["low"], df["close"],
                length=self.kc_period, scalar=self.kc_atr_multiple,
            )

            if bbands is None or kc is None:
                return False

            bb_upper_col = [c for c in bbands.columns if c.startswith("BBU")]
            bb_lower_col = [c for c in bbands.columns if c.startswith("BBL")]
            kc_upper_col = [c for c in kc.columns if c.startswith("KCU")]
            kc_lower_col = [c for c in kc.columns if c.startswith("KCL")]

            if not all([bb_upper_col, bb_lower_col, kc_upper_col, kc_lower_col]):
                return False

            bb_u = bbands[bb_upper_col[0]].iloc[-1]
            bb_l = bbands[bb_lower_col[0]].iloc[-1]
            kc_u = kc[kc_upper_col[0]].iloc[-1]
            kc_l = kc[kc_lower_col[0]].iloc[-1]

            if any(pd.isna(v) for v in [bb_u, bb_l, kc_u, kc_l]):
                return False

            # Squeeze: BB inside KC
            return bb_u < kc_u and bb_l > kc_l

        except Exception as e:
            logger.error("squeeze_calc_error", error=str(e))
            return False
