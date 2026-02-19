"""Backtesting engine.

Replays historical bars through the complete pipeline:
  bars -> IndicatorCalculator -> Strategy -> SignalGenerator -> Risk -> PaperExecutor
"""

from __future__ import annotations

from src.backtesting.results import BacktestResults
from src.core.logging import get_logger
from src.core.models import Bar, RiskDecision
from src.execution.order_manager import OrderManager
from src.execution.paper_executor import PaperExecutor
from src.indicators.calculator import IndicatorCalculator
from src.risk.manager import RiskManager
from src.signals.generator import SignalGenerator
from src.strategies.base import BaseStrategy

logger = get_logger("backtesting")


class BacktestEngine:
    """Replays historical bars through the trading pipeline."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        starting_equity: float = 10000.0,
        risk_config: dict | None = None,
    ) -> None:
        self.strategies = strategies
        self.starting_equity = starting_equity
        self.risk_config = risk_config

    def run(self, bars: list[Bar]) -> BacktestResults:
        """Execute a backtest over the given bars.

        Creates fresh instances of all components for isolation.
        """
        if not bars:
            return BacktestResults(
                strategy_name=self.strategies[0].name if self.strategies else "unknown",
                starting_equity=self.starting_equity,
                ending_equity=self.starting_equity,
            )

        # Build pipeline components fresh for each run
        risk_mgr = RiskManager(
            account_equity=self.starting_equity,
            risk_config=self.risk_config,
        )
        signal_gen = SignalGenerator(
            strategies=self.strategies,
            risk_manager=risk_mgr,
            sqlite_engine=None,
        )
        executor = PaperExecutor(
            paper_mode=True,
            slippage_ticks_mean=0.25,
            slippage_ticks_std=0.1,
        )
        order_mgr = OrderManager(
            executor=executor,
            risk_manager=risk_mgr,
        )
        indicator_calc = IndicatorCalculator()

        results = BacktestResults(
            strategy_name=self.strategies[0].name if self.strategies else "unknown",
            start_date=str(bars[0].timestamp),
            end_date=str(bars[-1].timestamp),
            starting_equity=self.starting_equity,
            ending_equity=self.starting_equity,
        )

        for bar in bars:
            # Compute indicators
            snapshot = indicator_calc.update(bar)

            # Generate and evaluate signals
            risk_results = signal_gen.on_bar(bar, snapshot)
            results.signals_generated += len(risk_results)
            results.signals_rejected += sum(
                1 for r in risk_results if r.decision == RiskDecision.REJECTED
            )

            # Process approved signals into positions
            order_mgr.process_signals(risk_results)

            # Check open positions for exits
            closed_trades = order_mgr.on_bar(bar)
            results.trades.extend(closed_trades)

            # Update unrealized P&L
            unrealized = order_mgr.get_total_unrealized_pnl()
            risk_mgr.daily_tracker.update_unrealized(unrealized)

            results.bars_processed += 1

        # Force-close any remaining open positions at last bar's close
        if order_mgr.open_positions:
            remaining = order_mgr.force_close_all(bars[-1].close)
            results.trades.extend(remaining)

        results.compute_metrics()

        logger.info(
            "backtest_complete",
            bars=results.bars_processed,
            trades=results.metrics.total_trades if results.metrics else 0,
            net_pnl=results.metrics.net_pnl if results.metrics else 0,
        )

        return results
