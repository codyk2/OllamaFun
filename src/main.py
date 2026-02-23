"""Application entry point.

Wires together all modules: Tradovate data, indicators, risk, strategies,
signals, multi-account execution, journal, LLM, scheduling, and dashboard.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from src.config import MES_SPEC, settings
from src.core.account_manager import AccountManager
from src.core.database import (
    get_duckdb_connection,
    get_sqlite_engine,
    init_duckdb,
    init_sqlite_db,
)
from src.core.logging import get_logger, setup_logging
from src.core.models import Bar, EquitySnapshot, RiskDecision
from src.execution.base_executor import BaseExecutor
from src.execution.copy_trader import CopyTradeManager
from src.execution.order_manager import OrderManager
from src.execution.paper_executor import PaperExecutor
from src.indicators.calculator import IndicatorCalculator
from src.journal.recorder import TradeRecorder
from src.llm.client import OllamaClient
from src.market_data.aggregator import BarAggregator
from src.market_data.tradovate_provider import TradovateProvider
from src.monitoring.health import HealthMonitor
from src.risk.apex_drawdown import ApexDrawdownTracker
from src.risk.manager import RiskManager
from src.scheduler.scheduler import TradingScheduler
from src.signals.generator import SignalGenerator
from src.strategies.mean_reversion import MeanReversionStrategy
from src.tradovate.auth import TradovateAuth
from src.tradovate.rest_client import TradovateRestClient

logger = get_logger("main")


class TradingApp:
    """Main application orchestrator with multi-account prop firm support."""

    def __init__(self) -> None:
        # Core infrastructure
        self.sqlite_engine = get_sqlite_engine()
        self.duckdb_conn = get_duckdb_connection()
        self.health = HealthMonitor()

        # Load accounts
        config_path = Path(settings.trading.account_config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent.parent / config_path
        self.account_mgr = AccountManager(config_path)
        self.accounts = self.account_mgr.load_accounts()
        logger.info("accounts_loaded", count=len(self.accounts),
                     ids=[a.account_id for a in self.accounts])

        # Tradovate auth + REST client (shared across accounts)
        self.tv_auth = TradovateAuth(settings.tradovate)
        self.tv_rest = TradovateRestClient(self.tv_auth, settings.tradovate)

        # Market data provider (ONE connection, shared)
        self.tv_provider = TradovateProvider(
            auth=self.tv_auth,
            config=settings.tradovate,
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

        # Strategy
        self.strategy = MeanReversionStrategy()

        # Journal
        self.trade_recorder = TradeRecorder(sqlite_engine=self.sqlite_engine)

        # Per-account: risk managers, executors, order managers
        self.risk_managers: dict[str, RiskManager] = {}
        self.executors: dict[str, BaseExecutor] = {}
        self.order_managers: dict[str, OrderManager] = {}

        for acct in self.accounts:
            # Apex drawdown tracker
            apex_dd = ApexDrawdownTracker(
                account_id=acct.account_id,
                trailing_threshold=acct.trailing_drawdown,
                max_equity_high=acct.max_equity_high,
                current_equity=acct.equity,
                starting_equity=acct.equity,
                profit_goal=acct.profit_goal,
            )

            # Per-account risk manager
            rm = RiskManager(
                account_equity=acct.equity,
                apex_drawdown=apex_dd,
                max_contracts=acct.max_contracts,
            )
            self.risk_managers[acct.account_id] = rm

            # Per-account executor (paper or live)
            if settings.trading.paper_mode:
                executor = PaperExecutor(
                    account_id=acct.account_id,
                    commission=MES_SPEC["commission_per_side"] * 2,
                )
            else:
                from src.execution.tradovate_executor import TradovateExecutor
                executor = TradovateExecutor(
                    rest_client=self.tv_rest,
                    tradovate_account_id=acct.tradovate_account_id or 0,
                    account_id=acct.account_id,
                    contract_id=0,  # Resolved after connect
                )
            self.executors[acct.account_id] = executor

            # Per-account order manager
            om = OrderManager(
                executor=executor,
                risk_manager=rm,
                recorder=self.trade_recorder,
                account_id=acct.account_id,
            )
            self.order_managers[acct.account_id] = om

        # Copy trade manager
        self.copy_trader = CopyTradeManager(
            account_managers=self.order_managers,
        )

        # Signal generator (multi-account mode: no single risk_manager)
        self.signal_generator = SignalGenerator(
            strategies=[self.strategy],
            risk_manager=None,
            sqlite_engine=self.sqlite_engine,
        )

        # LLM
        self.ollama_client = OllamaClient()

        # Scheduler (multi-account daily trackers)
        daily_trackers = {
            aid: rm.daily_tracker
            for aid, rm in self.risk_managers.items()
        }
        self.scheduler = TradingScheduler(
            daily_trackers=daily_trackers,
            health_monitor=self.health,
            trade_recorder=self.trade_recorder,
            equity_getter=self._get_equity_snapshots,
        )

        # Dashboard subprocess
        self._dashboard_proc: subprocess.Popen | None = None

    async def start(self) -> None:
        """Initialize and start all systems."""
        setup_logging()
        logger.info("app_starting", paper_mode=settings.trading.paper_mode,
                     accounts=len(self.accounts))

        # Initialize databases
        init_sqlite_db(self.sqlite_engine)
        init_duckdb(self.duckdb_conn)
        logger.info("databases_initialized")

        # Register health monitors
        self.health.register_broker(self.tv_provider)
        self.health.register_duckdb(self.duckdb_conn)
        self.health.register_sqlite(self.sqlite_engine)

        # Check LLM availability
        llm_available = await self.ollama_client.is_available()
        logger.info("llm_status", available=llm_available)

        # Start scheduler
        self.scheduler.start()

        # Connect to Tradovate
        connected = await self.tv_provider.connect()
        if connected:
            await self.tv_provider.subscribe_realtime_bars()
            logger.info("tradovate_data_streaming")
        else:
            logger.warning("tradovate_not_connected", msg="Running without live data")

        # Start dashboard
        self._start_dashboard()

        # Run health check
        status = self.health.check_all()
        logger.info("health_check_complete", status=status.overall_status.value)

        # Main event loop
        logger.info("app_started", msg="Trading assistant is running")
        try:
            await self.tv_provider.run_forever()
        except KeyboardInterrupt:
            logger.info("app_shutdown_requested")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shut down all systems."""
        logger.info("app_shutting_down")

        # Force-close any open positions on all accounts
        for account_id, om in self.order_managers.items():
            if om.open_positions:
                logger.warning("force_closing_positions",
                               account=account_id,
                               count=len(om.open_positions))
                om.force_close_all(0.0)

        # Generate final daily summaries for all accounts
        for account_id in self.risk_managers:
            self.trade_recorder.generate_daily_summary(account_id=account_id)

        # Stop scheduler
        self.scheduler.stop()

        # Flush any pending bars
        self.aggregator.flush()

        # Disconnect Tradovate
        await self.tv_provider.disconnect()
        await self.tv_auth.close()

        # Stop dashboard
        if self._dashboard_proc:
            self._dashboard_proc.terminate()

        # Close DB connections
        self.duckdb_conn.close()
        self.sqlite_engine.dispose()

        logger.info("app_shutdown_complete")

    def _on_5s_bar(self, bar: Bar) -> None:
        """Handle incoming 5-second bar from Tradovate."""
        self.aggregator.on_bar(bar)

    def _on_1m_bar(self, bar: Bar) -> None:
        """Handle completed 1-minute bar — the main trading pipeline.

        Multi-account flow:
        1. Compute indicators
        2. Generate raw signals from strategies
        3. Evaluate each signal against each account's risk manager
        4. Execute approved signals via per-account order managers
        5. Check all positions for exits
        6. Update drawdown tracking per account
        """
        # 1. Compute indicators
        snapshot = self.indicator_calc.update(bar)
        if snapshot:
            self._store_indicator_snapshot(snapshot)

        # 2. Generate raw signals
        signals = self.signal_generator.generate_signals(bar, snapshot)

        # 3. Evaluate each signal against each account's risk manager
        for signal in signals:
            atr = snapshot.atr_14 or 3.0 if snapshot else 3.0
            results = self.signal_generator.evaluate_signal_for_accounts(
                signal,
                atr=atr,
                risk_managers=self.risk_managers,
                current_time=bar.timestamp,
                trading_window=self.strategy.config.trading_window,
            )

            # 4. Execute approved signals
            for account_id, result in results.items():
                if result.decision == RiskDecision.APPROVED:
                    self.order_managers[account_id].process_signals([result])

        # 5. Check all accounts for exits
        all_closed = self.copy_trader.on_bar(bar)

        # 6. Update equity and drawdown tracking per account
        for account_id, rm in self.risk_managers.items():
            om = self.order_managers[account_id]
            unrealized = om.get_total_unrealized_pnl()
            rm.daily_tracker.update_unrealized(unrealized)

            # Update Apex drawdown tracking
            if rm.apex_drawdown:
                realized = rm.daily_tracker.realized_pnl_today
                acct = self.account_mgr.get_account(account_id)
                if acct:
                    current_equity = acct.equity + realized + unrealized
                    rm.apex_drawdown.update_equity(current_equity)

        # 7. AI review for closed trades
        for account_id, trades in all_closed.items():
            for trade in trades:
                asyncio.create_task(self._ai_review_trade(trade))

    def _on_5m_bar(self, bar: Bar) -> None:
        """Handle completed 5-minute bar."""
        logger.info("bar_5m", close=bar.close, volume=bar.volume)

    async def _ai_review_trade(self, trade) -> None:
        """Request AI review for a closed trade."""
        try:
            review = await self.ollama_client.review_trade(trade)
            trade.ai_review = review
            logger.info("ai_review_complete", account=trade.account_id,
                        trade_pnl=trade.pnl_dollars)
        except Exception as e:
            logger.error("ai_review_failed", error=str(e))

    def _get_equity_snapshots(self) -> list[EquitySnapshot]:
        """Build equity snapshots for all accounts."""
        snapshots = []
        for account_id, rm in self.risk_managers.items():
            om = self.order_managers[account_id]
            unrealized = om.get_total_unrealized_pnl()
            realized = rm.daily_tracker.realized_pnl_today
            acct = self.account_mgr.get_account(account_id)
            base_equity = acct.equity if acct else 50000.0

            snapshots.append(EquitySnapshot(
                account_id=account_id,
                equity=base_equity + realized + unrealized,
                unrealized_pnl=unrealized,
                realized_pnl_today=realized,
            ))
        return snapshots

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
