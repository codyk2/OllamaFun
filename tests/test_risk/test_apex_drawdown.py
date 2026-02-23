"""Tests for ApexDrawdownTracker -- Apex Trader Funding trailing drawdown.

Covers:
    1. Initial state (drawdown_floor, drawdown_remaining, drawdown_used_pct)
    2. update_equity() ratchets high water mark UP only
    3. update_equity() with declining equity does NOT change max_equity_high
    4. drawdown_floor calculation (max_equity_high - trailing_threshold)
    5. Account busts when equity drops below drawdown_floor
    6. can_trade() returns True when healthy
    7. can_trade() returns False when account_busted
    8. can_trade() returns False when within safety_margin (>=90% used)
    9. Properties at various equity levels
   10. Edge case: equity exactly at floor
"""

from __future__ import annotations

import pytest

from src.risk.apex_drawdown import ApexDrawdownTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(
    *,
    account_id: str = "APEX-TEST",
    trailing_threshold: float = 3_000.0,
    starting_equity: float = 50_000.0,
) -> ApexDrawdownTracker:
    """Build a default tracker for a $50K Apex account with $3K drawdown."""
    return ApexDrawdownTracker(
        account_id=account_id,
        trailing_threshold=trailing_threshold,
        max_equity_high=starting_equity,
        current_equity=starting_equity,
    )


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    """After construction the tracker should reflect a fresh account."""

    def test_initial_drawdown_floor(self):
        t = _make_tracker()
        assert t.drawdown_floor == 47_000.0

    def test_initial_drawdown_remaining(self):
        t = _make_tracker()
        # Equity is at 50K, floor is 47K => 3K remaining
        assert t.drawdown_remaining == 3_000.0

    def test_initial_drawdown_used_pct_zero(self):
        t = _make_tracker()
        assert t.drawdown_used_pct == pytest.approx(0.0)

    def test_initial_not_busted(self):
        t = _make_tracker()
        assert t.account_busted is False

    def test_initial_can_trade(self):
        t = _make_tracker()
        ok, reason = t.can_trade()
        assert ok is True
        assert reason == ""


# ---------------------------------------------------------------------------
# 2. update_equity() ratchets high water mark UP only
# ---------------------------------------------------------------------------

class TestHighWaterMarkRatchet:
    """max_equity_high must only increase, never decrease."""

    def test_higher_equity_raises_hwm(self):
        t = _make_tracker()
        t.update_equity(51_000.0)
        assert t.max_equity_high == 51_000.0

    def test_multiple_increases_ratchet(self):
        t = _make_tracker()
        for eq in [50_500.0, 51_000.0, 51_500.0, 52_000.0]:
            t.update_equity(eq)
        assert t.max_equity_high == 52_000.0

    def test_floor_moves_up_with_hwm(self):
        t = _make_tracker()
        t.update_equity(52_000.0)
        # floor should be 52K - 3K = 49K
        assert t.drawdown_floor == 49_000.0


# ---------------------------------------------------------------------------
# 3. Declining equity does NOT change max_equity_high
# ---------------------------------------------------------------------------

class TestDecliningEquity:
    """A drawdown should NOT lower the high water mark or the floor."""

    def test_decline_does_not_lower_hwm(self):
        t = _make_tracker()
        t.update_equity(49_000.0)
        assert t.max_equity_high == 50_000.0

    def test_decline_then_partial_recovery(self):
        t = _make_tracker()
        t.update_equity(48_000.0)
        t.update_equity(49_500.0)
        # HWM should still be the original 50K
        assert t.max_equity_high == 50_000.0

    def test_floor_unchanged_on_decline(self):
        t = _make_tracker()
        t.update_equity(49_000.0)
        assert t.drawdown_floor == 47_000.0

    def test_current_equity_still_updated_on_decline(self):
        t = _make_tracker()
        t.update_equity(48_500.0)
        assert t.current_equity == 48_500.0


# ---------------------------------------------------------------------------
# 4. drawdown_floor calculation
# ---------------------------------------------------------------------------

class TestDrawdownFloor:
    """Floor = max_equity_high - trailing_threshold at every point."""

    def test_floor_at_start(self):
        t = _make_tracker(starting_equity=50_000.0, trailing_threshold=3_000.0)
        assert t.drawdown_floor == 47_000.0

    def test_floor_after_profit(self):
        t = _make_tracker()
        t.update_equity(53_000.0)
        assert t.drawdown_floor == 50_000.0

    def test_floor_with_different_threshold(self):
        t = _make_tracker(trailing_threshold=2_500.0, starting_equity=25_000.0)
        assert t.drawdown_floor == 22_500.0

    def test_floor_with_large_threshold(self):
        t = _make_tracker(trailing_threshold=5_000.0, starting_equity=100_000.0)
        assert t.drawdown_floor == 95_000.0


# ---------------------------------------------------------------------------
# 5. Account busts when equity drops below drawdown_floor
# ---------------------------------------------------------------------------

class TestAccountBust:
    """Account should bust when equity <= floor."""

    def test_equity_below_floor_busts(self):
        t = _make_tracker()  # floor = 47K
        t.update_equity(46_999.0)
        assert t.account_busted is True

    def test_large_drop_busts(self):
        t = _make_tracker()
        t.update_equity(40_000.0)
        assert t.account_busted is True

    def test_bust_is_permanent(self):
        """Once busted, recovery does not un-bust the account."""
        t = _make_tracker()
        t.update_equity(46_000.0)
        assert t.account_busted is True
        # "Recovery" -- but bust flag stays True
        t.update_equity(55_000.0)
        assert t.account_busted is True

    def test_bust_after_profit_then_crash(self):
        t = _make_tracker()
        t.update_equity(53_000.0)  # floor -> 50K
        t.update_equity(49_999.0)  # below new floor of 50K
        assert t.account_busted is True


# ---------------------------------------------------------------------------
# 6. can_trade() returns True when healthy
# ---------------------------------------------------------------------------

class TestCanTradeHealthy:

    def test_fresh_account(self):
        t = _make_tracker()
        ok, reason = t.can_trade()
        assert ok is True
        assert reason == ""

    def test_after_small_drawdown(self):
        t = _make_tracker()
        t.update_equity(49_000.0)  # used 1K of 3K -> 33%
        ok, reason = t.can_trade()
        assert ok is True

    def test_after_profit(self):
        t = _make_tracker()
        t.update_equity(55_000.0)
        ok, reason = t.can_trade()
        assert ok is True

    def test_at_89_percent_used(self):
        """Just under the 90% threshold -- still tradeable."""
        t = _make_tracker()
        # 89% of 3K used = 2670 used => equity at 50K - 2670 = 47330
        # drawdown_remaining = 47330 - 47000 = 330
        # used_pct = 1 - (330 / 3000) = 0.89
        t.update_equity(47_330.0)
        assert t.drawdown_used_pct == pytest.approx(0.89, abs=1e-9)
        ok, _ = t.can_trade()
        assert ok is True


# ---------------------------------------------------------------------------
# 7. can_trade() returns False when account_busted
# ---------------------------------------------------------------------------

class TestCanTradeBusted:

    def test_busted_blocks_trading(self):
        t = _make_tracker()
        t.update_equity(46_000.0)
        ok, reason = t.can_trade()
        assert ok is False
        assert "busted" in reason.lower()

    def test_busted_reason_includes_account_id(self):
        t = _make_tracker(account_id="APEX-123")
        t.update_equity(46_000.0)
        _, reason = t.can_trade()
        assert "APEX-123" in reason

    def test_busted_reason_includes_equity_and_floor(self):
        t = _make_tracker()
        t.update_equity(46_000.0)
        _, reason = t.can_trade()
        assert "46,000.00" in reason
        assert "47,000.00" in reason


# ---------------------------------------------------------------------------
# 8. can_trade() returns False in warn zone (>=90% used)
# ---------------------------------------------------------------------------

class TestCanTradeWarnZone:

    def test_at_90_percent_used(self):
        t = _make_tracker()
        # 90% of 3K used = 2700 used => equity = 50K - 2700 = 47300
        # remaining = 47300 - 47000 = 300, used_pct = 1 - 300/3000 = 0.90
        t.update_equity(47_300.0)
        assert t.drawdown_used_pct == pytest.approx(0.90)
        ok, reason = t.can_trade()
        assert ok is False
        assert "safety" in reason.lower()

    def test_at_95_percent_used(self):
        t = _make_tracker()
        # 95% of 3K = 2850 used => equity = 50K - 2850 = 47150
        t.update_equity(47_150.0)
        assert t.drawdown_used_pct == pytest.approx(0.95)
        ok, reason = t.can_trade()
        assert ok is False

    def test_warn_zone_reason_includes_remaining(self):
        t = _make_tracker()
        t.update_equity(47_200.0)  # remaining = 200
        _, reason = t.can_trade()
        assert "200.00" in reason

    def test_warn_zone_reason_includes_pct(self):
        t = _make_tracker()
        t.update_equity(47_150.0)
        _, reason = t.can_trade()
        # 95.0% should appear in reason
        assert "95.0%" in reason


# ---------------------------------------------------------------------------
# 9. Properties at various equity levels
# ---------------------------------------------------------------------------

class TestProperties:

    def test_drawdown_remaining_after_profit(self):
        t = _make_tracker()
        t.update_equity(52_000.0)
        # floor = 49K, remaining = 52K - 49K = 3K (full cushion restored)
        assert t.drawdown_remaining == pytest.approx(3_000.0)

    def test_drawdown_remaining_after_loss(self):
        t = _make_tracker()
        t.update_equity(48_500.0)
        # floor still 47K, remaining = 48500 - 47000 = 1500
        assert t.drawdown_remaining == pytest.approx(1_500.0)

    def test_drawdown_used_pct_half(self):
        t = _make_tracker()
        t.update_equity(48_500.0)
        # remaining = 1500, used_pct = 1 - 1500/3000 = 0.50
        assert t.drawdown_used_pct == pytest.approx(0.50)

    def test_drawdown_used_pct_zero_after_new_high(self):
        t = _make_tracker()
        t.update_equity(55_000.0)
        assert t.drawdown_used_pct == pytest.approx(0.0)

    def test_drawdown_used_pct_with_zero_threshold(self):
        """Edge: trailing_threshold == 0 should return 1.0 (fully used)."""
        t = ApexDrawdownTracker(
            account_id="ZERO",
            trailing_threshold=0.0,
            max_equity_high=50_000.0,
            current_equity=50_000.0,
        )
        assert t.drawdown_used_pct == 1.0

    def test_drawdown_remaining_negative_when_below_floor(self):
        """Once below the floor, remaining goes negative."""
        t = _make_tracker()
        t.update_equity(46_500.0)
        assert t.drawdown_remaining == pytest.approx(-500.0)

    def test_drawdown_used_pct_exceeds_one_when_below_floor(self):
        t = _make_tracker()
        t.update_equity(46_500.0)
        # remaining = -500, used_pct = 1 - (-500/3000) = 1.1667
        assert t.drawdown_used_pct > 1.0


# ---------------------------------------------------------------------------
# 10. Edge case: equity exactly at floor
# ---------------------------------------------------------------------------

class TestEquityExactlyAtFloor:

    def test_equity_equal_to_floor_busts(self):
        """Apex rule: equity <= floor means bust (not just strictly less)."""
        t = _make_tracker()  # floor = 47K
        t.update_equity(47_000.0)
        assert t.account_busted is True

    def test_equity_equal_to_floor_blocks_trading(self):
        t = _make_tracker()
        t.update_equity(47_000.0)
        ok, reason = t.can_trade()
        assert ok is False
        assert "busted" in reason.lower()

    def test_equity_one_cent_above_floor(self):
        """One cent above the floor is NOT busted (but likely in warn zone)."""
        t = _make_tracker()
        t.update_equity(47_000.01)
        assert t.account_busted is False

    def test_equity_at_floor_after_hwm_ratchet(self):
        t = _make_tracker()
        t.update_equity(53_000.0)  # floor -> 50K
        t.update_equity(50_000.0)  # exactly at new floor
        assert t.account_busted is True

    def test_drawdown_remaining_zero_at_floor(self):
        t = _make_tracker()
        # Don't use update_equity because it would bust.
        # Set current_equity directly to inspect the property.
        t.current_equity = 47_000.0
        assert t.drawdown_remaining == pytest.approx(0.0)

    def test_drawdown_used_pct_one_at_floor(self):
        t = _make_tracker()
        t.current_equity = 47_000.0
        assert t.drawdown_used_pct == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration-style: full lifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    """Simulate a realistic trading session from open to bust."""

    def test_profit_then_drawdown_to_bust(self):
        t = _make_tracker(starting_equity=50_000.0, trailing_threshold=3_000.0)

        # Morning profits
        t.update_equity(50_500.0)
        t.update_equity(51_200.0)
        t.update_equity(51_800.0)
        assert t.max_equity_high == 51_800.0
        assert t.drawdown_floor == 48_800.0
        ok, _ = t.can_trade()
        assert ok is True

        # Afternoon losses
        t.update_equity(50_000.0)
        assert t.max_equity_high == 51_800.0  # unchanged
        ok, _ = t.can_trade()
        assert ok is True

        # Getting close to the limit
        t.update_equity(49_100.0)
        # remaining = 49100 - 48800 = 300, used_pct = 1 - 300/3000 = 0.90
        ok, reason = t.can_trade()
        assert ok is False
        assert "safety" in reason.lower()

        # Account bust
        t.update_equity(48_700.0)
        assert t.account_busted is True
        ok, reason = t.can_trade()
        assert ok is False
        assert "busted" in reason.lower()

    def test_gradual_climb_keeps_floor_tight(self):
        """Each new high moves the floor up, shrinking the safety net below old equity."""
        t = _make_tracker(starting_equity=50_000.0, trailing_threshold=2_500.0)

        t.update_equity(50_100.0)
        assert t.drawdown_floor == pytest.approx(47_600.0)

        t.update_equity(50_200.0)
        assert t.drawdown_floor == pytest.approx(47_700.0)

        t.update_equity(50_300.0)
        assert t.drawdown_floor == pytest.approx(47_800.0)

        # Now a drop back to 48K -- still above floor (47800) but floor has
        # ratcheted 800 above the original 47000
        t.update_equity(48_000.0)
        assert t.account_busted is False
        assert t.drawdown_remaining == pytest.approx(200.0)
