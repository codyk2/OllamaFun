"""Tests for MeanReversionStrategy."""

from datetime import UTC, datetime

import pytest

from src.core.models import Bar, Direction, IndicatorSnapshot
from src.strategies.base import StrategyConfig
from src.strategies.mean_reversion import DEFAULT_PARAMS, MeanReversionStrategy


def _make_bar(close=5000.0):
    return Bar(
        timestamp=datetime.now(UTC),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1000,
    )


def _make_snapshot(
    close=5000.0,
    bb_lower=4990.0,
    bb_upper=5010.0,
    bb_middle=5000.0,
    rsi=50.0,
    atr=3.0,
    vwap=5000.0,
    ema_9=4999.0,
    ema_21=5001.0,
    keltner_lower=4985.0,
    keltner_upper=5015.0,
):
    return IndicatorSnapshot(
        timestamp=datetime.now(UTC),
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        rsi_14=rsi,
        atr_14=atr,
        vwap=vwap,
        ema_9=ema_9,
        ema_21=ema_21,
        keltner_upper=keltner_upper,
        keltner_middle=5000.0,
        keltner_lower=keltner_lower,
    )


class TestMeanReversionLong:
    def test_long_signal_on_bb_lower_touch_and_rsi_oversold(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=30.0, vwap=5000.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.entry_price == 4990.0
        assert signal.stop_loss < 4990.0
        assert signal.take_profit > 4990.0

    def test_no_signal_when_rsi_not_oversold(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=50.0)
        signal = strat.on_bar(bar, snap)
        assert signal is None

    def test_no_signal_when_bb_not_touched(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4995.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=30.0)
        signal = strat.on_bar(bar, snap)
        assert signal is None

    def test_keltner_filter_blocks_extreme_trend(self):
        """If price is below keltner_lower, don't go long (extreme downtrend)."""
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4988.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=25.0, keltner_lower=4992.0)
        signal = strat.on_bar(bar, snap)
        assert signal is None  # close=4988 < keltner_lower=4992

    def test_keltner_filter_disabled(self):
        config = StrategyConfig(
            name="test", params={"require_keltner_filter": False}
        )
        strat = MeanReversionStrategy(config)
        bar = _make_bar(close=4988.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=25.0, keltner_lower=4992.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None  # Keltner filter off


class TestMeanReversionShort:
    def test_short_signal_on_bb_upper_touch_and_rsi_overbought(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=5010.0)
        snap = _make_snapshot(bb_upper=5010.0, rsi=70.0, vwap=5000.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.direction == Direction.SHORT
        assert signal.entry_price == 5010.0
        assert signal.stop_loss > 5010.0
        assert signal.take_profit < 5010.0

    def test_no_signal_when_rsi_not_overbought(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=5010.0)
        snap = _make_snapshot(bb_upper=5010.0, rsi=55.0)
        signal = strat.on_bar(bar, snap)
        assert signal is None

    def test_keltner_filter_blocks_extreme_uptrend(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=5012.0)
        snap = _make_snapshot(bb_upper=5010.0, rsi=75.0, keltner_upper=5008.0)
        signal = strat.on_bar(bar, snap)
        assert signal is None  # close=5012 > keltner_upper=5008


class TestConfidence:
    def test_base_confidence_is_0_5(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=34.0, vwap=None, ema_9=None, ema_21=None)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.confidence == pytest.approx(0.5)

    def test_rsi_extreme_bonus(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=20.0, vwap=None, ema_9=None, ema_21=None)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.confidence >= 0.65  # 0.5 + 0.15

    def test_vwap_alignment_bonus(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=34.0, vwap=5000.0, ema_9=None, ema_21=None)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.confidence >= 0.6  # 0.5 + 0.10

    def test_ema_alignment_bonus(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(
            bb_lower=4990.0, rsi=34.0, vwap=None, ema_9=4998.0, ema_21=5002.0
        )
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.confidence >= 0.6  # 0.5 + 0.10

    def test_max_confidence_capped_at_1(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4985.0)
        snap = _make_snapshot(
            bb_lower=4990.0,
            rsi=20.0,
            vwap=5005.0,
            ema_9=4995.0,
            ema_21=5005.0,
            atr=3.0,
            keltner_lower=4980.0,  # Below close so Keltner filter passes
        )
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.confidence <= 1.0


class TestStopAndTarget:
    def test_stop_below_entry_for_long(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=30.0, atr=3.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.stop_loss < signal.entry_price

    def test_target_above_entry_for_long(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=30.0, atr=3.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.take_profit > signal.entry_price

    def test_stop_above_entry_for_short(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=5010.0)
        snap = _make_snapshot(bb_upper=5010.0, rsi=70.0, atr=3.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.stop_loss > signal.entry_price


class TestEdgeCases:
    def test_min_atr_filter(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=30.0, atr=0.3)
        signal = strat.on_bar(bar, snap)
        assert signal is None

    def test_missing_indicators_returns_none(self):
        strat = MeanReversionStrategy()
        bar = _make_bar()
        snap = IndicatorSnapshot(timestamp=datetime.now(UTC))  # All None
        signal = strat.on_bar(bar, snap)
        assert signal is None

    def test_default_config(self):
        strat = MeanReversionStrategy()
        assert strat.name == "mean_reversion"
        assert strat.params["rsi_oversold"] == 35.0
        assert strat.params["rsi_overbought"] == 65.0

    def test_custom_config_merges_defaults(self):
        config = StrategyConfig(name="custom_mr", params={"rsi_oversold": 30.0})
        strat = MeanReversionStrategy(config)
        assert strat.params["rsi_oversold"] == 30.0
        assert strat.params["rsi_overbought"] == 65.0  # From defaults

    def test_invalid_params_raises(self):
        with pytest.raises(AssertionError):
            config = StrategyConfig(name="bad", params={"rsi_oversold": 60.0})
            MeanReversionStrategy(config)

    def test_signal_has_market_context(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=30.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert signal.market_context is not None

    def test_signal_reason_string(self):
        strat = MeanReversionStrategy()
        bar = _make_bar(close=4990.0)
        snap = _make_snapshot(bb_lower=4990.0, rsi=30.0)
        signal = strat.on_bar(bar, snap)
        assert signal is not None
        assert "BB lower" in signal.reason
        assert "RSI oversold" in signal.reason
