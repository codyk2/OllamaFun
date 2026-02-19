"""Trade recording and daily summary generation."""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.database import (
    DailySummaryRow,
    EquitySnapshotRow,
    RiskEventRow,
    TradeRow,
    get_session,
)
from src.core.logging import get_logger
from src.core.models import (
    Direction,
    EquitySnapshot,
    RiskEvent,
    Trade,
    TradeStatus,
)

logger = get_logger("journal.recorder")


class TradeRecorder:
    """Records trades and generates daily summaries."""

    def __init__(self, sqlite_engine=None) -> None:
        self.sqlite_engine = sqlite_engine
        self._today_trades: list[Trade] = []

    def record_trade(self, trade: Trade) -> int | None:
        """Persist a closed trade to SQLite. Returns the trade ID."""
        self._today_trades.append(trade)

        if self.sqlite_engine is None:
            return None

        try:
            session = get_session(self.sqlite_engine)
            row = self._trade_to_row(trade)
            session.add(row)
            session.commit()
            trade_id = row.id
            session.close()
            logger.info("trade_recorded", trade_id=trade_id, pnl=trade.pnl_dollars)
            return trade_id
        except Exception as e:
            logger.error("trade_record_failed", error=str(e))
            return None

    def record_risk_event(self, event: RiskEvent) -> None:
        """Persist a risk event."""
        if self.sqlite_engine is None:
            return

        try:
            session = get_session(self.sqlite_engine)
            row = RiskEventRow(
                event_type=event.event_type,
                details=event.details,
                severity=event.severity.value,
                created_at=event.timestamp,
            )
            session.add(row)
            session.commit()
            session.close()
        except Exception as e:
            logger.error("risk_event_record_failed", error=str(e))

    def record_equity_snapshot(self, snapshot: EquitySnapshot) -> None:
        """Persist an equity snapshot."""
        if self.sqlite_engine is None:
            return

        try:
            session = get_session(self.sqlite_engine)
            row = EquitySnapshotRow(
                equity=snapshot.equity,
                unrealized_pnl=snapshot.unrealized_pnl,
                realized_pnl_today=snapshot.realized_pnl_today,
                snapshot_time=snapshot.timestamp,
            )
            session.add(row)
            session.commit()
            session.close()
        except Exception as e:
            logger.error("equity_snapshot_failed", error=str(e))

    def generate_daily_summary(self, date: str | None = None) -> DailySummaryRow | None:
        """Generate and persist a daily summary from today's trades."""
        if date is None:
            date = datetime.now(UTC).strftime("%Y-%m-%d")

        trades = self._today_trades
        if not trades:
            return None

        closed = [t for t in trades if t.status == TradeStatus.CLOSED and t.pnl_dollars is not None]
        if not closed:
            return None

        winners = [t for t in closed if t.pnl_dollars > 0]
        losers = [t for t in closed if t.pnl_dollars < 0]

        gross_pnl = sum(t.pnl_dollars for t in closed)
        net_pnl = gross_pnl  # Commission already included in pnl_dollars

        avg_winner = sum(t.pnl_dollars for t in winners) / len(winners) if winners else 0.0
        avg_loser = sum(t.pnl_dollars for t in losers) / len(losers) if losers else 0.0

        gross_loss = abs(sum(t.pnl_dollars for t in losers))
        profit_factor = (sum(t.pnl_dollars for t in winners) / gross_loss) if gross_loss > 0 else 0.0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in closed:
            cumulative += t.pnl_dollars
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        row = DailySummaryRow(
            date=date,
            total_trades=len(closed),
            winners=len(winners),
            losers=len(losers),
            gross_pnl=sum(t.pnl_dollars for t in winners) + sum(t.pnl_dollars for t in losers),
            net_pnl=net_pnl,
            max_drawdown=max_dd,
            win_rate=len(winners) / len(closed) if closed else 0.0,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            profit_factor=profit_factor,
        )

        if self.sqlite_engine is not None:
            try:
                session = get_session(self.sqlite_engine)
                session.merge(row)
                session.commit()
                session.close()
            except Exception as e:
                logger.error("daily_summary_failed", error=str(e))

        return row

    def get_trades_for_date(self, date: str) -> list[Trade]:
        """Load all trades for a given date from SQLite."""
        if self.sqlite_engine is None:
            return [
                t for t in self._today_trades
                if t.entry_time and t.entry_time.strftime("%Y-%m-%d") == date
            ]

        try:
            session = get_session(self.sqlite_engine)
            rows = (
                session.query(TradeRow)
                .filter(TradeRow.entry_time.like(f"{date}%"))
                .all()
            )
            trades = [self._row_to_trade(r) for r in rows]
            session.close()
            return trades
        except Exception as e:
            logger.error("load_trades_failed", error=str(e))
            return []

    def reset_daily(self) -> None:
        """Clear in-memory daily trade buffer."""
        self._today_trades.clear()

    def _trade_to_row(self, trade: Trade) -> TradeRow:
        return TradeRow(
            strategy=trade.strategy,
            symbol=trade.symbol,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            quantity=trade.quantity,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            status=trade.status.value,
            pnl_ticks=trade.pnl_ticks,
            pnl_dollars=trade.pnl_dollars,
            risk_reward_actual=trade.risk_reward_actual,
            commission=trade.commission,
            slippage_ticks=trade.slippage_ticks,
            signal_confidence=trade.signal_confidence,
            ai_review=trade.ai_review,
            notes=trade.notes,
        )

    def _row_to_trade(self, row: TradeRow) -> Trade:
        return Trade(
            id=row.id,
            strategy=row.strategy,
            symbol=row.symbol,
            direction=Direction(row.direction),
            entry_price=row.entry_price,
            exit_price=row.exit_price,
            stop_loss=row.stop_loss,
            take_profit=row.take_profit,
            quantity=row.quantity,
            entry_time=row.entry_time,
            exit_time=row.exit_time,
            status=TradeStatus(row.status),
            pnl_ticks=row.pnl_ticks,
            pnl_dollars=row.pnl_dollars,
            risk_reward_actual=row.risk_reward_actual,
            commission=row.commission,
            slippage_ticks=row.slippage_ticks,
            signal_confidence=row.signal_confidence,
            ai_review=row.ai_review,
            notes=row.notes,
        )
