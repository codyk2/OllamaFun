"""Technical indicator calculations using pandas-ta.

Computes VWAP, Bollinger Bands, Keltner Channels, RSI, ATR, and EMAs
from OHLCV bar data. Results are returned as IndicatorSnapshot objects.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.core.logging import get_logger
from src.core.models import Bar, IndicatorSnapshot

logger = get_logger("indicators")


class IndicatorCalculator:
    """Computes technical indicators from a rolling window of bars."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_atr_multiple: float = 1.5,
        rsi_period: int = 14,
        atr_period: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_multiple = kc_atr_multiple
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

        # Rolling bar buffer (keep enough for longest lookback + padding)
        self._max_bars = max(bb_period, kc_period, rsi_period, atr_period, ema_slow) + 50
        self._bars: list[Bar] = []

        # VWAP state (resets each session)
        self._vwap_cum_vol: float = 0.0
        self._vwap_cum_tp_vol: float = 0.0
        self._vwap_session_date: str | None = None

    def update(self, bar: Bar) -> IndicatorSnapshot | None:
        """Add a new bar and compute all indicators.

        Returns None if not enough data yet.
        """
        self._bars.append(bar)
        if len(self._bars) > self._max_bars:
            self._bars = self._bars[-self._max_bars:]

        # Need minimum bars for any calculation
        min_required = max(self.bb_period, self.rsi_period, self.atr_period) + 1
        if len(self._bars) < min_required:
            return None

        df = self._to_dataframe()

        snapshot = IndicatorSnapshot(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            vwap=self._compute_vwap(bar),
            **self._compute_bbands(df),
            **self._compute_keltner(df),
            rsi_14=self._compute_rsi(df),
            atr_14=self._compute_atr(df),
            ema_9=self._compute_ema(df, self.ema_fast),
            ema_21=self._compute_ema(df, self.ema_slow),
        )

        return snapshot

    def reset_vwap(self) -> None:
        """Reset VWAP for a new session."""
        self._vwap_cum_vol = 0.0
        self._vwap_cum_tp_vol = 0.0
        self._vwap_session_date = None

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert bar buffer to a pandas DataFrame."""
        data = {
            "open": [b.open for b in self._bars],
            "high": [b.high for b in self._bars],
            "low": [b.low for b in self._bars],
            "close": [b.close for b in self._bars],
            "volume": [b.volume for b in self._bars],
        }
        return pd.DataFrame(data)

    def _compute_vwap(self, bar: Bar) -> float | None:
        """Compute session-anchored VWAP."""
        session_date = bar.timestamp.strftime("%Y-%m-%d")

        # Reset on new session
        if self._vwap_session_date != session_date:
            self._vwap_cum_vol = 0.0
            self._vwap_cum_tp_vol = 0.0
            self._vwap_session_date = session_date

        typical_price = (bar.high + bar.low + bar.close) / 3
        self._vwap_cum_tp_vol += typical_price * bar.volume
        self._vwap_cum_vol += bar.volume

        if self._vwap_cum_vol == 0:
            return None

        return self._vwap_cum_tp_vol / self._vwap_cum_vol

    def _compute_bbands(self, df: pd.DataFrame) -> dict:
        """Compute Bollinger Bands."""
        try:
            bbands = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
            if bbands is None or bbands.empty:
                return {"bb_upper": None, "bb_middle": None, "bb_lower": None}

            cols = bbands.columns.tolist()
            # pandas-ta names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
            lower_col = [c for c in cols if c.startswith("BBL")]
            mid_col = [c for c in cols if c.startswith("BBM")]
            upper_col = [c for c in cols if c.startswith("BBU")]

            return {
                "bb_lower": _last_value(bbands[lower_col[0]]) if lower_col else None,
                "bb_middle": _last_value(bbands[mid_col[0]]) if mid_col else None,
                "bb_upper": _last_value(bbands[upper_col[0]]) if upper_col else None,
            }
        except Exception as e:
            logger.error("bbands_calc_error", error=str(e))
            return {"bb_upper": None, "bb_middle": None, "bb_lower": None}

    def _compute_keltner(self, df: pd.DataFrame) -> dict:
        """Compute Keltner Channels."""
        try:
            kc = ta.kc(
                df["high"], df["low"], df["close"],
                length=self.kc_period,
                scalar=self.kc_atr_multiple,
            )
            if kc is None or kc.empty:
                return {"keltner_upper": None, "keltner_middle": None, "keltner_lower": None}

            cols = kc.columns.tolist()
            lower_col = [c for c in cols if c.startswith("KCL")]
            mid_col = [c for c in cols if c.startswith("KCB")]
            upper_col = [c for c in cols if c.startswith("KCU")]

            return {
                "keltner_lower": _last_value(kc[lower_col[0]]) if lower_col else None,
                "keltner_middle": _last_value(kc[mid_col[0]]) if mid_col else None,
                "keltner_upper": _last_value(kc[upper_col[0]]) if upper_col else None,
            }
        except Exception as e:
            logger.error("keltner_calc_error", error=str(e))
            return {"keltner_upper": None, "keltner_middle": None, "keltner_lower": None}

    def _compute_rsi(self, df: pd.DataFrame) -> float | None:
        """Compute RSI."""
        try:
            rsi = ta.rsi(df["close"], length=self.rsi_period)
            if rsi is None:
                return None
            return _last_value(rsi)
        except Exception as e:
            logger.error("rsi_calc_error", error=str(e))
            return None

    def _compute_atr(self, df: pd.DataFrame) -> float | None:
        """Compute ATR."""
        try:
            atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
            if atr is None:
                return None
            return _last_value(atr)
        except Exception as e:
            logger.error("atr_calc_error", error=str(e))
            return None

    def _compute_ema(self, df: pd.DataFrame, period: int) -> float | None:
        """Compute EMA."""
        try:
            ema = ta.ema(df["close"], length=period)
            if ema is None:
                return None
            return _last_value(ema)
        except Exception as e:
            logger.error("ema_calc_error", error=str(e), period=period)
            return None


def _last_value(series: pd.Series) -> float | None:
    """Get the last non-NaN value from a pandas Series."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)
