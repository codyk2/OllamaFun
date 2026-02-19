"""Application entry point.

Wires together all modules: IB data, indicators, risk, strategies,
signals, execution, journal, LLM, scheduling, and dashboard.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from src.config import settings
from src.core.database import (
    get_duckdb_connection,
    get_sqlite_engine,
    init_duckdb,
    init_sqlite_db,
)
from src.core.logging import get_logger, setup_logging
from src.core.models import Bar, EquitySnapshot
from src.execution.order_manager import OrderManager
from src.execution.paper_executor import PaperExecutor
from src.indicators.calculator import IndicatorCalculator
from src.journal.recorder import TradeRecorder
from src.llm.client import OllamaClient
from src.market_data.aggregator import BarAggregator
from src.market_data.ib_provider import IBProvider
from src.monitoring.health import HealthMonitor
from src.risk.manager import RiskManager
from src.scheduler.scheduler import TradingScheduler
from src.signals.generator import SignalGenerator
from src.strategies.mean_reversion import MeanReversionStrategy

logger = get_logger("main")


class TradingApp:
    """Main application orchestrator."""

    def __init__(self) -> None:
        # Core infrastructure
        self.sqlite_engine = get_sqlite_engine()
        self.duckdb_conn = get_duckdb_connection()
        self.health = HealthMonitor()

        # Market data
        self.ib_provider = IBProvider(
            on_bar=self._on_5s_bar,
        )

        # Indicators
        self.indicator_calc = IndicatorCalculator()

        # Aggregator: 5s bars -> 1m/5m bars
        self.aggregator = BarAggregator(
            duckdb_conn=self.duckdb_conn,
            on_1m_bar=self._on_1m_bar,
            on_5m_bar=self._on_5m_bar,
        )

        # Risk
        self.risk_manager = RiskManager(
            account_equity=settings.trading.account_equity,
        )

        # Strategy
        self.strategy = MeanReversionStrategy()

        # Journal
        self.trade_recorder = TradeRecorder(sqlite_engine=self.sqlite_engine)

        # Signals
        self.signal_generator = SignalGenerator(
            strategies=[self.strategy],
            risk_manager=self.risk_manager,
            sqlite_engine=self.sqlite_engine,
        )

        # Execution
        self.executor = PaperExecutor()
        self.order_manager = OrderManager(
            executor=self.executor,
            risk_manager=self.risk_manager,
            recorder=self.trade_recorder,
        )

        # LLM
        self.ollama_client = OllamaClient()

        # Scheduler
        self.scheduler = TradingScheduler(
            daily_tracker=self.risk_manager.daily_tracker,
            health_monitor=self.health,
            trade_recorder=self.trade_recorder,
            equity_getter=self._get_equity_snapshot,
        )

        # Dashboard subprocess
        self._dashboard_proc: subprocess.Popen | None = None

    async def start(self) -> None:
        """Initialize and start all systems."""
        setup_logging()
        logger.info("app_starting", paper_mode=settings.trading.paper_mode)

        # Initialize databases
        init_sqlite_db(self.sqlite_engine)
        init_duckdb(self.duckdb_conn)
        logger.info("databases_initialized")

        # Register health monitors
        self.health.register_ib(self.ib_provider)
        self.health.register_duckdb(self.duckdb_conn)
        self.health.register_sqlite(self.sqlite_engine)

        # Check LLM availability
        llm_available = await self.ollama_client.is_available()
        logger.info("llm_status", available=llm_available)

        # Start scheduler
        self.scheduler.start()

        # Connect to IB
        connected = await self.ib_provider.connect()
        if connected:
            await self.ib_provider.subscribe_realtime_bars()
            logger.info("ib_data_streaming")
        else:
            logger.warning("ib_not_connected", msg="Running without live data")

        # Start dashboard
        self._start_dashboard()

        # Run health check
        status = self.health.check_all()
        logger.info("health_check_complete", status=status.overall_status.value)

        # Main event loop
        logger.info("app_started", msg="Trading assistant is running")
        try:
            await self.ib_provider.run_forever()
        except KeyboardInterrupt:
            logger.info("app_shutdown_requested")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shut down all systems."""
        logger.info("app_shutting_down")

        # Force-close any open positions
        if self.order_manager.open_positions:
            logger.warning("force_closing_positions", count=len(self.order_manager.open_positions))
            self.order_manager.force_close_all(0.0)  # Will use last known price

        # Generate final daily summary
        self.trade_recorder.generate_daily_summary()

        # Stop scheduler
        self.scheduler.stop()

        # Flush any pending bars
        self.aggregator.flush()

        # Disconnect IB
        await self.ib_provider.disconnect()

        # Stop dashboard
        if self._dashboard_proc:
            self._dashboard_proc.terminate()

        # Close DB connections
        self.duckdb_conn.close()
        self.sqlite_engine.dispose()

        logger.info("app_shutdown_complete")

    def _on_5s_bar(self, bar: Bar) -> None:
        """Handle incoming 5-second bar from IB."""
        self.aggregator.on_bar(bar)

    def _on_1m_bar(self, bar: Bar) -> None:
        """Handle completed 1-minute bar — the main trading pipeline."""
        # 1. Compute indicators
        snapshot = self.indicator_calc.update(bar)
        if snapshot:
            self._store_indicator_snapshot(snapshot)

        # 2. Generate and evaluate signals
        risk_results = self.signal_generator.on_bar(bar, snapshot)

        # 3. Execute approved signals
        self.order_manager.process_signals(risk_results)

        # 4. Check open positions for exits
        closed_trades = self.order_manager.on_bar(bar)

        # 5. Update unrealized P&L in daily tracker
        unrealized = self.order_manager.get_total_unrealized_pnl()
        self.risk_manager.daily_tracker.update_unrealized(unrealized)

        # 6. AI review for closed trades (fire and forget)
        for trade in closed_trades:
            asyncio.create_task(self._ai_review_trade(trade))

    def _on_5m_bar(self, bar: Bar) -> None:
        """Handle completed 5-minute bar."""
        logger.info("bar_5m", close=bar.close, volume=bar.volume)

    async def _ai_review_trade(self, trade) -> None:
        """Request AI review for a closed trade."""
        try:
            review = await self.ollama_client.review_trade(trade)
            trade.ai_review = review
            logger.info("ai_review_complete", trade_pnl=trade.pnl_dollars)
        except Exception as e:
            logger.error("ai_review_failed", error=str(e))

    def _get_equity_snapshot(self) -> EquitySnapshot:
        """Build current equity snapshot for scheduler."""
        unrealized = self.order_manager.get_total_unrealized_pnl()
        realized = self.risk_manager.daily_tracker.realized_pnl_today
        return EquitySnapshot(
            equity=settings.trading.account_equity + realized + unrealized,
            unrealized_pnl=unrealized,
            realized_pnl_today=realized,
        )

    def _store_indicator_snapshot(self, snapshot) -> None:
        """Store computed indicators in DuckDB."""
        try:
            self.duckdb_conn.execute(
                """INSERT OR REPLACE INTO indicator_cache
                   (timestamp, symbol, timeframe, vwap, bb_upper, bb_middle, bb_lower,
                    keltner_upper, keltner_middle, keltner_lower, rsi_14, atr_14,
                    ema_9, ema_21, volume_profile_poc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    snapshot.timestamp, snapshot.symbol, snapshot.timeframe,
                    snapshot.vwap, snapshot.bb_upper, snapshot.bb_middle, snapshot.bb_lower,
                    snapshot.keltner_upper, snapshot.keltner_middle, snapshot.keltner_lower,
                    snapshot.rsi_14, snapshot.atr_14, snapshot.ema_9, snapshot.ema_21,
                    snapshot.volume_profile_poc,
                ],
            )
        except Exception as e:
            logger.error("indicator_store_failed", error=str(e))

    def _start_dashboard(self) -> None:
        """Launch Streamlit dashboard in a subprocess."""
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        port = settings.dashboard.streamlit_port

        try:
            self._dashboard_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run",
                    str(dashboard_path),
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--theme.base", "dark",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("dashboard_started", port=port)
        except Exception as e:
            logger.error("dashboard_start_failed", error=str(e))


def main():
    """Entry point."""
    app = TradingApp()
    asyncio.run(app.start())


if __name__ == "__main__":
    main()
