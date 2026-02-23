"""Tests for the CopyTradeManager class.

Covers initialization, on_bar delegation, process_signals fan-out,
force_close_all propagation, and get_total_unrealized_pnl aggregation
across multiple mocked OrderManager instances.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, call

import pytest

from src.core.models import (
    Bar,
    Direction,
    RiskDecision,
    RiskResult,
    Signal,
    Trade,
    TradeStatus,
)
from src.execution.copy_trader import CopyTradeManager
from src.execution.order_manager import OrderManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(close: float = 5000.0) -> Bar:
    """Create a minimal Bar for testing."""
    return Bar(
        timestamp=datetime.now(UTC),
        open=close - 1.0,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=100,
    )


def _make_trade(
    account_id: str = "acct1",
    direction: Direction = Direction.LONG,
    entry: float = 5000.0,
    exit_price: float | None = None,
    status: TradeStatus = TradeStatus.OPEN,
    pnl: float | None = None,
) -> Trade:
    """Create a Trade for testing."""
    return Trade(
        account_id=account_id,
        strategy="mean_reversion",
        direction=direction,
        entry_price=entry,
        exit_price=exit_price,
        stop_loss=entry - 4.0 if direction == Direction.LONG else entry + 4.0,
        take_profit=entry + 8.0 if direction == Direction.LONG else entry - 8.0,
        quantity=1,
        entry_time=datetime.now(UTC),
        exit_time=datetime.now(UTC) if exit_price else None,
        status=status,
        pnl_dollars=pnl,
    )


def _make_risk_result(
    decision: RiskDecision = RiskDecision.APPROVED,
    account_id: str = "acct1",
) -> RiskResult:
    """Create a RiskResult for testing."""
    signal = Signal(
        strategy="mean_reversion",
        direction=Direction.LONG,
        confidence=0.75,
        entry_price=5000.0,
        stop_loss=4996.0,
        take_profit=5008.0,
    )
    return RiskResult(
        decision=decision,
        position_size=1,
        reason="approved" if decision == RiskDecision.APPROVED else "rejected",
        signal=signal,
        account_id=account_id,
    )


def _mock_order_manager(account_id: str = "acct1") -> MagicMock:
    """Create a MagicMock with spec=OrderManager."""
    mock = MagicMock(spec=OrderManager)
    mock.account_id = account_id
    # Defaults: no closed trades, no unrealized PnL, no signals processed
    mock.on_bar.return_value = []
    mock.force_close_all.return_value = []
    mock.get_total_unrealized_pnl.return_value = 0.0
    mock.process_signals.return_value = []
    return mock


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestCopyTradeManagerInit:
    """Tests for CopyTradeManager.__init__."""

    def test_init_with_single_account(self):
        mgr1 = _mock_order_manager("acct1")
        ctm = CopyTradeManager({"acct1": mgr1})

        assert len(ctm.account_managers) == 1
        assert ctm.account_managers["acct1"] is mgr1

    def test_init_with_multiple_accounts(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        mgr3 = _mock_order_manager("acct3")

        ctm = CopyTradeManager({
            "acct1": mgr1,
            "acct2": mgr2,
            "acct3": mgr3,
        })

        assert len(ctm.account_managers) == 3
        assert set(ctm.account_managers.keys()) == {"acct1", "acct2", "acct3"}

    def test_init_with_empty_dict(self):
        ctm = CopyTradeManager({})
        assert len(ctm.account_managers) == 0


# ---------------------------------------------------------------------------
# on_bar
# ---------------------------------------------------------------------------

class TestOnBar:
    """Tests for CopyTradeManager.on_bar."""

    def test_on_bar_delegates_to_all_managers(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        bar = _make_bar(5000.0)
        ctm.on_bar(bar)

        mgr1.on_bar.assert_called_once_with(bar)
        mgr2.on_bar.assert_called_once_with(bar)

    def test_on_bar_returns_dict_of_closed_trades(self):
        trade1 = _make_trade("acct1", exit_price=5008.0, status=TradeStatus.CLOSED, pnl=8.96)
        trade2 = _make_trade("acct2", exit_price=4996.0, status=TradeStatus.CLOSED, pnl=-6.04)

        mgr1 = _mock_order_manager("acct1")
        mgr1.on_bar.return_value = [trade1]
        mgr2 = _mock_order_manager("acct2")
        mgr2.on_bar.return_value = [trade2]

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        bar = _make_bar(5000.0)
        result = ctm.on_bar(bar)

        assert "acct1" in result
        assert "acct2" in result
        assert result["acct1"] == [trade1]
        assert result["acct2"] == [trade2]

    def test_on_bar_returns_empty_lists_when_no_exits(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        bar = _make_bar(5000.0)
        result = ctm.on_bar(bar)

        assert result == {"acct1": [], "acct2": []}

    def test_on_bar_multiple_closed_trades_per_account(self):
        trade1 = _make_trade("acct1", exit_price=5008.0, status=TradeStatus.CLOSED, pnl=8.96)
        trade2 = _make_trade("acct1", exit_price=5004.0, status=TradeStatus.CLOSED, pnl=3.96)

        mgr1 = _mock_order_manager("acct1")
        mgr1.on_bar.return_value = [trade1, trade2]

        ctm = CopyTradeManager({"acct1": mgr1})
        bar = _make_bar(5010.0)
        result = ctm.on_bar(bar)

        assert len(result["acct1"]) == 2

    def test_on_bar_exception_in_one_manager_does_not_affect_others(self):
        """If one OrderManager.on_bar raises, CopyTradeManager should catch it
        and still call the remaining managers."""
        mgr1 = _mock_order_manager("acct1")
        mgr1.on_bar.side_effect = RuntimeError("IB connection dropped")

        trade2 = _make_trade("acct2", exit_price=5008.0, status=TradeStatus.CLOSED, pnl=8.96)
        mgr2 = _mock_order_manager("acct2")
        mgr2.on_bar.return_value = [trade2]

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        bar = _make_bar(5000.0)
        result = ctm.on_bar(bar)

        # acct1 should have empty list due to exception, acct2 should work fine
        assert result["acct1"] == []
        assert result["acct2"] == [trade2]

    def test_on_bar_with_empty_account_managers(self):
        ctm = CopyTradeManager({})
        bar = _make_bar(5000.0)
        result = ctm.on_bar(bar)
        assert result == {}

    def test_on_bar_passes_same_bar_to_all_managers(self):
        """Ensure the exact same Bar object is sent to every manager."""
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        mgr3 = _mock_order_manager("acct3")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2, "acct3": mgr3})

        bar = _make_bar(5050.0)
        ctm.on_bar(bar)

        for mgr in [mgr1, mgr2, mgr3]:
            args, _ = mgr.on_bar.call_args
            assert args[0] is bar


# ---------------------------------------------------------------------------
# force_close_all
# ---------------------------------------------------------------------------

class TestForceCloseAll:
    """Tests for CopyTradeManager.force_close_all."""

    def test_force_close_all_calls_each_manager(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        ctm.force_close_all(5000.0)

        mgr1.force_close_all.assert_called_once_with(5000.0)
        mgr2.force_close_all.assert_called_once_with(5000.0)

    def test_force_close_all_returns_closed_trades_per_account(self):
        trade1 = _make_trade("acct1", exit_price=5000.0, status=TradeStatus.CLOSED, pnl=-1.04)
        trade2 = _make_trade("acct2", exit_price=5000.0, status=TradeStatus.CLOSED, pnl=3.96)

        mgr1 = _mock_order_manager("acct1")
        mgr1.force_close_all.return_value = [trade1]
        mgr2 = _mock_order_manager("acct2")
        mgr2.force_close_all.return_value = [trade2]

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        result = ctm.force_close_all(5000.0)

        assert result["acct1"] == [trade1]
        assert result["acct2"] == [trade2]

    def test_force_close_all_returns_empty_when_no_positions(self):
        mgr1 = _mock_order_manager("acct1")
        mgr1.force_close_all.return_value = []
        ctm = CopyTradeManager({"acct1": mgr1})

        result = ctm.force_close_all(5000.0)
        assert result == {"acct1": []}

    def test_force_close_all_exception_in_one_manager_does_not_affect_others(self):
        mgr1 = _mock_order_manager("acct1")
        mgr1.force_close_all.side_effect = RuntimeError("executor timeout")

        trade2 = _make_trade("acct2", exit_price=5000.0, status=TradeStatus.CLOSED, pnl=3.96)
        mgr2 = _mock_order_manager("acct2")
        mgr2.force_close_all.return_value = [trade2]

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        result = ctm.force_close_all(5000.0)

        assert result["acct1"] == []
        assert result["acct2"] == [trade2]

    def test_force_close_all_multiple_trades_per_account(self):
        trade1 = _make_trade("acct1", entry=5000.0, exit_price=4990.0, status=TradeStatus.CLOSED)
        trade2 = _make_trade("acct1", entry=5010.0, exit_price=4990.0, status=TradeStatus.CLOSED)

        mgr1 = _mock_order_manager("acct1")
        mgr1.force_close_all.return_value = [trade1, trade2]

        ctm = CopyTradeManager({"acct1": mgr1})
        result = ctm.force_close_all(4990.0)

        assert len(result["acct1"]) == 2

    def test_force_close_all_passes_correct_price(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        ctm.force_close_all(4999.75)

        mgr1.force_close_all.assert_called_once_with(4999.75)
        mgr2.force_close_all.assert_called_once_with(4999.75)

    def test_force_close_all_with_empty_account_managers(self):
        ctm = CopyTradeManager({})
        result = ctm.force_close_all(5000.0)
        assert result == {}


# ---------------------------------------------------------------------------
# get_total_unrealized_pnl
# ---------------------------------------------------------------------------

class TestGetTotalUnrealizedPnl:
    """Tests for CopyTradeManager.get_total_unrealized_pnl."""

    def test_aggregates_across_all_accounts(self):
        mgr1 = _mock_order_manager("acct1")
        mgr1.get_total_unrealized_pnl.return_value = 125.50
        mgr2 = _mock_order_manager("acct2")
        mgr2.get_total_unrealized_pnl.return_value = -30.25

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        result = ctm.get_total_unrealized_pnl()

        assert result == {"acct1": 125.50, "acct2": -30.25}

    def test_returns_per_account_pnl_not_summed(self):
        """The method returns a dict of per-account P&L, not a single sum."""
        mgr1 = _mock_order_manager("acct1")
        mgr1.get_total_unrealized_pnl.return_value = 100.0
        mgr2 = _mock_order_manager("acct2")
        mgr2.get_total_unrealized_pnl.return_value = 200.0

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        result = ctm.get_total_unrealized_pnl()

        assert isinstance(result, dict)
        assert result["acct1"] == 100.0
        assert result["acct2"] == 200.0

    def test_all_accounts_zero_pnl(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        result = ctm.get_total_unrealized_pnl()

        assert result == {"acct1": 0.0, "acct2": 0.0}

    def test_exception_in_one_manager_returns_zero_for_that_account(self):
        mgr1 = _mock_order_manager("acct1")
        mgr1.get_total_unrealized_pnl.side_effect = RuntimeError("data error")

        mgr2 = _mock_order_manager("acct2")
        mgr2.get_total_unrealized_pnl.return_value = 50.0

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        result = ctm.get_total_unrealized_pnl()

        assert result["acct1"] == 0.0
        assert result["acct2"] == 50.0

    def test_calls_get_total_unrealized_pnl_on_each_manager(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        mgr3 = _mock_order_manager("acct3")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2, "acct3": mgr3})

        ctm.get_total_unrealized_pnl()

        mgr1.get_total_unrealized_pnl.assert_called_once()
        mgr2.get_total_unrealized_pnl.assert_called_once()
        mgr3.get_total_unrealized_pnl.assert_called_once()

    def test_negative_pnl_values(self):
        mgr1 = _mock_order_manager("acct1")
        mgr1.get_total_unrealized_pnl.return_value = -500.0
        mgr2 = _mock_order_manager("acct2")
        mgr2.get_total_unrealized_pnl.return_value = -250.75

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})
        result = ctm.get_total_unrealized_pnl()

        assert result["acct1"] == -500.0
        assert result["acct2"] == -250.75

    def test_empty_account_managers(self):
        ctm = CopyTradeManager({})
        result = ctm.get_total_unrealized_pnl()
        assert result == {}


# ---------------------------------------------------------------------------
# process_signals
# ---------------------------------------------------------------------------

class TestProcessSignals:
    """Tests for CopyTradeManager.process_signals."""

    def test_fans_out_signals_to_correct_accounts(self):
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        rr1 = _make_risk_result(RiskDecision.APPROVED, "acct1")
        rr2 = _make_risk_result(RiskDecision.APPROVED, "acct2")

        ctm.process_signals({"acct1": [rr1], "acct2": [rr2]})

        mgr1.process_signals.assert_called_once_with([rr1])
        mgr2.process_signals.assert_called_once_with([rr2])

    def test_returns_trades_per_account(self):
        trade1 = _make_trade("acct1")
        trade2 = _make_trade("acct2")

        mgr1 = _mock_order_manager("acct1")
        mgr1.process_signals.return_value = [trade1]
        mgr2 = _mock_order_manager("acct2")
        mgr2.process_signals.return_value = [trade2]

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        rr1 = _make_risk_result(RiskDecision.APPROVED, "acct1")
        rr2 = _make_risk_result(RiskDecision.APPROVED, "acct2")

        result = ctm.process_signals({"acct1": [rr1], "acct2": [rr2]})

        assert result["acct1"] == [trade1]
        assert result["acct2"] == [trade2]

    def test_unknown_account_is_skipped(self):
        mgr1 = _mock_order_manager("acct1")
        ctm = CopyTradeManager({"acct1": mgr1})

        rr_unknown = _make_risk_result(RiskDecision.APPROVED, "acct_unknown")
        result = ctm.process_signals({"acct_unknown": [rr_unknown]})

        # Unknown account should not appear in result, and mgr1 should not be called
        assert "acct_unknown" not in result
        mgr1.process_signals.assert_not_called()

    def test_process_signals_exception_returns_empty_list_for_account(self):
        mgr1 = _mock_order_manager("acct1")
        mgr1.process_signals.side_effect = RuntimeError("executor error")

        trade2 = _make_trade("acct2")
        mgr2 = _mock_order_manager("acct2")
        mgr2.process_signals.return_value = [trade2]

        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        rr1 = _make_risk_result(RiskDecision.APPROVED, "acct1")
        rr2 = _make_risk_result(RiskDecision.APPROVED, "acct2")

        result = ctm.process_signals({"acct1": [rr1], "acct2": [rr2]})

        assert result["acct1"] == []
        assert result["acct2"] == [trade2]

    def test_process_signals_empty_input(self):
        mgr1 = _mock_order_manager("acct1")
        ctm = CopyTradeManager({"acct1": mgr1})

        result = ctm.process_signals({})
        assert result == {}
        mgr1.process_signals.assert_not_called()

    def test_process_signals_multiple_risk_results_per_account(self):
        trade1 = _make_trade("acct1", entry=5000.0)
        trade2 = _make_trade("acct1", entry=5010.0)

        mgr1 = _mock_order_manager("acct1")
        mgr1.process_signals.return_value = [trade1, trade2]

        ctm = CopyTradeManager({"acct1": mgr1})

        rr1 = _make_risk_result(RiskDecision.APPROVED, "acct1")
        rr2 = _make_risk_result(RiskDecision.APPROVED, "acct1")

        result = ctm.process_signals({"acct1": [rr1, rr2]})

        assert len(result["acct1"]) == 2
        # Verify the manager received both risk results together
        mgr1.process_signals.assert_called_once_with([rr1, rr2])

    def test_process_signals_no_trades_returned(self):
        mgr1 = _mock_order_manager("acct1")
        mgr1.process_signals.return_value = []

        ctm = CopyTradeManager({"acct1": mgr1})

        rr1 = _make_risk_result(RiskDecision.REJECTED, "acct1")
        result = ctm.process_signals({"acct1": [rr1]})

        assert result["acct1"] == []

    def test_process_signals_partial_account_coverage(self):
        """Signals only for acct1, acct2 should not be called."""
        mgr1 = _mock_order_manager("acct1")
        mgr2 = _mock_order_manager("acct2")
        ctm = CopyTradeManager({"acct1": mgr1, "acct2": mgr2})

        rr1 = _make_risk_result(RiskDecision.APPROVED, "acct1")
        result = ctm.process_signals({"acct1": [rr1]})

        mgr1.process_signals.assert_called_once()
        mgr2.process_signals.assert_not_called()
        assert "acct2" not in result


# ---------------------------------------------------------------------------
# Integration-style: combining methods
# ---------------------------------------------------------------------------

class TestCopyTradeManagerIntegration:
    """End-to-end-style tests combining multiple CopyTradeManager methods."""

    def test_full_lifecycle_signals_then_bar_then_close(self):
        """Process signals -> on_bar (no exit) -> force_close_all."""
        trade_open = _make_trade("acct1")
        trade_closed = _make_trade(
            "acct1",
            exit_price=5000.0,
            status=TradeStatus.CLOSED,
            pnl=-1.04,
        )

        mgr1 = _mock_order_manager("acct1")
        mgr1.process_signals.return_value = [trade_open]
        mgr1.on_bar.return_value = []
        mgr1.force_close_all.return_value = [trade_closed]
        mgr1.get_total_unrealized_pnl.return_value = 12.50

        ctm = CopyTradeManager({"acct1": mgr1})

        # 1. Process signals
        rr = _make_risk_result(RiskDecision.APPROVED, "acct1")
        trades = ctm.process_signals({"acct1": [rr]})
        assert len(trades["acct1"]) == 1

        # 2. on_bar -- no exit
        bar = _make_bar(5002.0)
        closed = ctm.on_bar(bar)
        assert closed["acct1"] == []

        # 3. Check unrealized PnL
        pnl = ctm.get_total_unrealized_pnl()
        assert pnl["acct1"] == 12.50

        # 4. Force close
        result = ctm.force_close_all(5000.0)
        assert len(result["acct1"]) == 1
        assert result["acct1"][0].status == TradeStatus.CLOSED

    def test_many_accounts_different_outcomes(self):
        """Three accounts with different on_bar results."""
        trade_a = _make_trade("acct1", exit_price=5008.0, status=TradeStatus.CLOSED, pnl=8.96)
        trade_b = _make_trade("acct2", exit_price=4996.0, status=TradeStatus.CLOSED, pnl=-6.04)

        mgr1 = _mock_order_manager("acct1")
        mgr1.on_bar.return_value = [trade_a]
        mgr1.get_total_unrealized_pnl.return_value = 0.0

        mgr2 = _mock_order_manager("acct2")
        mgr2.on_bar.return_value = [trade_b]
        mgr2.get_total_unrealized_pnl.return_value = 0.0

        mgr3 = _mock_order_manager("acct3")
        mgr3.on_bar.return_value = []
        mgr3.get_total_unrealized_pnl.return_value = 75.00

        ctm = CopyTradeManager({
            "acct1": mgr1,
            "acct2": mgr2,
            "acct3": mgr3,
        })

        bar = _make_bar(5008.0)
        closed = ctm.on_bar(bar)

        assert len(closed["acct1"]) == 1
        assert len(closed["acct2"]) == 1
        assert len(closed["acct3"]) == 0

        pnl = ctm.get_total_unrealized_pnl()
        assert pnl["acct1"] == 0.0
        assert pnl["acct2"] == 0.0
        assert pnl["acct3"] == 75.00

    def test_exception_isolation_across_all_methods(self):
        """An account that fails on every call should not block others."""
        mgr_bad = _mock_order_manager("bad")
        mgr_bad.process_signals.side_effect = RuntimeError("broken")
        mgr_bad.on_bar.side_effect = RuntimeError("broken")
        mgr_bad.force_close_all.side_effect = RuntimeError("broken")
        mgr_bad.get_total_unrealized_pnl.side_effect = RuntimeError("broken")

        trade_ok = _make_trade("ok", exit_price=5005.0, status=TradeStatus.CLOSED, pnl=4.96)
        mgr_ok = _mock_order_manager("ok")
        mgr_ok.process_signals.return_value = [trade_ok]
        mgr_ok.on_bar.return_value = [trade_ok]
        mgr_ok.force_close_all.return_value = [trade_ok]
        mgr_ok.get_total_unrealized_pnl.return_value = 100.0

        ctm = CopyTradeManager({"bad": mgr_bad, "ok": mgr_ok})

        rr_bad = _make_risk_result(RiskDecision.APPROVED, "bad")
        rr_ok = _make_risk_result(RiskDecision.APPROVED, "ok")

        # process_signals
        sig_result = ctm.process_signals({"bad": [rr_bad], "ok": [rr_ok]})
        assert sig_result["bad"] == []
        assert sig_result["ok"] == [trade_ok]

        # on_bar
        bar = _make_bar(5005.0)
        bar_result = ctm.on_bar(bar)
        assert bar_result["bad"] == []
        assert bar_result["ok"] == [trade_ok]

        # force_close_all
        close_result = ctm.force_close_all(5005.0)
        assert close_result["bad"] == []
        assert close_result["ok"] == [trade_ok]

        # get_total_unrealized_pnl
        pnl_result = ctm.get_total_unrealized_pnl()
        assert pnl_result["bad"] == 0.0
        assert pnl_result["ok"] == 100.0
