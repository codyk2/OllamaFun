"""Tests for TradeRecorder."""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.database import Base, get_sqlite_engine, init_sqlite_db
from src.core.models import (
    Direction,
    EquitySnapshot,
    RiskEvent,
    Severity,
    Trade,
    TradeStatus,
)
from src.journal.recorder import TradeRecorder


@pytest.fixture
def sqlite_engine():
    """In-memory SQLite engine for testing."""
    engine = get_sqlite_engine(db_url="sqlite:///:memory:")
    init_sqlite_db(engine)
    return engine


@pytest.fixture
def recorder(sqlite_engine):
    return TradeRecorder(sqlite_engine=sqlite_engine)


def _make_trade(pnl: float = 50.0, offset_min: int = 0) -> Trade:
    entry_time = datetime(2024, 1, 15, 9, 30, tzinfo=UTC) + timedelta(minutes=offset_min)
    return Trade(
        strategy="mean_reversion",
        direction=Direction.LONG,
        entry_price=5000.0,
        exit_price=5000.0 + (pnl / 1.25) * 0.25,
        stop_loss=4996.0,
        take_profit=5008.0,
        quantity=1,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(minutes=15),
        status=TradeStatus.CLOSED,
        pnl_ticks=pnl / 1.25,
        pnl_dollars=pnl,
    )


class TestRecordTrade:
    def test_record_trade_returns_id(self, recorder):
        trade = _make_trade(50.0)
        trade_id = recorder.record_trade(trade)
        assert trade_id is not None
        assert trade_id > 0

    def test_record_multiple_trades(self, recorder):
        id1 = recorder.record_trade(_make_trade(50.0, 0))
        id2 = recorder.record_trade(_make_trade(-20.0, 20))
        assert id1 != id2

    def test_trade_in_memory_buffer(self, recorder):
        recorder.record_trade(_make_trade(50.0))
        assert len(recorder._today_trades) == 1

    def test_record_without_engine(self):
        recorder = TradeRecorder(sqlite_engine=None)
        trade_id = recorder.record_trade(_make_trade())
        assert trade_id is None
        assert len(recorder._today_trades) == 1


class TestRecordRiskEvent:
    def test_record_risk_event(self, recorder):
        event = RiskEvent(
            event_type="DAILY_LIMIT",
            details={"pnl": -300.0},
            severity=Severity.CRITICAL,
        )
        recorder.record_risk_event(event)  # Should not raise


class TestRecordEquitySnapshot:
    def test_record_equity_snapshot(self, recorder):
        snapshot = EquitySnapshot(
            equity=10050.0,
            unrealized_pnl=50.0,
            realized_pnl_today=100.0,
        )
        recorder.record_equity_snapshot(snapshot)  # Should not raise


class TestDailySummary:
    def test_generate_daily_summary(self, recorder):
        recorder.record_trade(_make_trade(50.0, 0))
        recorder.record_trade(_make_trade(-20.0, 20))
        recorder.record_trade(_make_trade(30.0, 40))

        summary = recorder.generate_daily_summary("2024-01-15")
        assert summary is not None
        assert summary.total_trades == 3
        assert summary.winners == 2
        assert summary.losers == 1
        assert summary.win_rate == pytest.approx(2 / 3)

    def test_no_trades_returns_none(self, recorder):
        summary = recorder.generate_daily_summary("2024-01-15")
        assert summary is None

    def test_max_drawdown_computed(self, recorder):
        recorder.record_trade(_make_trade(50.0, 0))
        recorder.record_trade(_make_trade(-30.0, 20))
        recorder.record_trade(_make_trade(-20.0, 40))

        summary = recorder.generate_daily_summary("2024-01-15")
        assert summary.max_drawdown == pytest.approx(50.0)


class TestGetTradesForDate:
    def test_get_trades_for_date(self, recorder):
        recorder.record_trade(_make_trade(50.0, 0))
        recorder.record_trade(_make_trade(-20.0, 20))

        trades = recorder.get_trades_for_date("2024-01-15")
        assert len(trades) == 2

    def test_get_trades_wrong_date(self, recorder):
        recorder.record_trade(_make_trade(50.0, 0))
        trades = recorder.get_trades_for_date("2024-01-16")
        assert len(trades) == 0


class TestResetDaily:
    def test_reset_clears_buffer(self, recorder):
        recorder.record_trade(_make_trade())
        assert len(recorder._today_trades) == 1
        recorder.reset_daily()
        assert len(recorder._today_trades) == 0
