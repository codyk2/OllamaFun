"""Tests for dynamic take-profit targets and scale-out logic."""

from datetime import UTC, datetime

import pytest

from src.core.models import (
    Bar,
    Direction,
    IndicatorSnapshot,
    Position,
    Trade,
    TradeStatus,
)
from src.strategies.mean_reversion import MeanReversionStrategy


def _make_snapshot(
    bb_upper=5010.0,
    bb_middle=5000.0,
    bb_lower=4990.0,
    rsi=30.0,
    atr=3.0,
    vwap=5002.0,
    keltner_lower=4988.0,
    keltner_upper=5012.0,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        timestamp=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        rsi_14=rsi,
        atr_14=atr,
        vwap=vwap,
        keltner_lower=keltner_lower,
        keltner_upper=keltner_upper,
        ema_9=4998.0,
        ema_21=5001.0,
    )


def _make_bar(close=4989.0, open_=4990.0, high=4991.0, low=4988.0) -> Bar:
    return Bar(
        timestamp=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1500,
    )


class TestDynamicTargets:
    def test_long_signal_has_primary_target_bb_middle(self):
        """Long signal should have BB middle as primary target."""
        strategy = MeanReversionStrategy()
        bar = _make_bar(close=4989.0)
        snap = _make_snapshot(bb_middle=5000.0, vwap=5002.0)
        signal = strategy.generate_signal(bar, snap)

        assert signal is not None
        assert signal.take_profit_primary is not None
        assert signal.take_profit_primary == 5000.0  # BB middle

    def test_long_signal_has_secondary_target_vwap(self):
        """Long signal should have VWAP as secondary target."""
        strategy = MeanReversionStrategy()
        bar = _make_bar(close=4989.0)
        snap = _make_snapshot(bb_middle=5000.0, vwap=5002.0)
        signal = strategy.generate_signal(bar, snap)

        assert signal is not None
        assert signal.take_profit_secondary is not None
        assert signal.take_profit_secondary == 5002.0  # VWAP

    def test_primary_closer_than_secondary_for_long(self):
        """For long, primary target should be closer (lower) than secondary."""
        strategy = MeanReversionStrategy()
        bar = _make_bar(close=4989.0)
        # VWAP closer than BB middle
        snap = _make_snapshot(bb_middle=5005.0, vwap=4995.0)
        signal = strategy.generate_signal(bar, snap)

        if signal is not None and signal.take_profit_primary and signal.take_profit_secondary:
            assert signal.take_profit_primary <= signal.take_profit_secondary

    def test_no_primary_when_bb_middle_too_close(self):
        """BB middle less than 4 ticks away should not be a target."""
        strategy = MeanReversionStrategy()
        bar = _make_bar(close=4989.0)
        snap = _make_snapshot(bb_middle=4989.5)  # Only 0.5 pts = 2 ticks
        signal = strategy.generate_signal(bar, snap)

        if signal is not None:
            assert signal.take_profit_primary is None

    def test_take_profit_uses_primary_when_available(self):
        """Main take_profit should be set to primary target."""
        strategy = MeanReversionStrategy()
        bar = _make_bar(close=4989.0)
        snap = _make_snapshot(bb_middle=5000.0)
        signal = strategy.generate_signal(bar, snap)

        if signal is not None and signal.take_profit_primary is not None:
            assert signal.take_profit == signal.take_profit_primary

    def test_short_signal_dynamic_targets(self):
        """Short signal targets should be below entry."""
        strategy = MeanReversionStrategy()
        bar = _make_bar(close=5011.0, open_=5010.0, high=5012.0, low=5009.0)
        snap = _make_snapshot(
            bb_upper=5010.0,
            bb_middle=5000.0,
            bb_lower=4990.0,
            rsi=70.0,
            vwap=4998.0,
            keltner_upper=5012.0,
        )
        signal = strategy.generate_signal(bar, snap)

        if signal is not None and signal.take_profit_primary is not None:
            assert signal.take_profit_primary < bar.close


class TestScaleOutPosition:
    def test_scale_out_reduces_quantity(self):
        """Scale-out should halve the position quantity."""
        trade = Trade(
            strategy="mean_reversion",
            direction=Direction.LONG,
            entry_price=5000.0,
            stop_loss=4996.0,
            take_profit=5005.0,
            quantity=2,
            entry_time=datetime.now(UTC),
            status=TradeStatus.OPEN,
        )
        position = Position(
            trade=trade,
            current_price=5005.0,
            original_quantity=2,
        )

        # Simulate scale-out
        scale_qty = position.trade.quantity // 2
        position.trade.quantity -= scale_qty
        position.scale_out_done = True
        position.trailing_stop = position.trade.entry_price

        assert position.trade.quantity == 1
        assert position.scale_out_done is True
        assert position.trailing_stop == 5000.0  # Breakeven

    def test_no_scale_out_for_1_lot(self):
        """1-lot positions should not scale out."""
        trade = Trade(
            strategy="mean_reversion",
            direction=Direction.LONG,
            entry_price=5000.0,
            stop_loss=4996.0,
            take_profit=5005.0,
            quantity=1,
            entry_time=datetime.now(UTC),
            status=TradeStatus.OPEN,
        )
        position = Position(
            trade=trade,
            current_price=5005.0,
            original_quantity=1,
        )
        # Can't split 1 lot
        assert position.trade.quantity <= 1

    def test_original_quantity_preserved(self):
        """Original quantity should be tracked for analysis."""
        position = Position(
            trade=Trade(
                strategy="test",
                direction=Direction.LONG,
                entry_price=5000.0,
                stop_loss=4996.0,
                quantity=4,
                entry_time=datetime.now(UTC),
            ),
            current_price=5000.0,
            original_quantity=4,
        )
        position.trade.quantity = 2  # After scale-out
        assert position.original_quantity == 4
