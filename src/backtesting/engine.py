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
from src.indicators.regime import RegimeDetector
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
        use_regime_detection: bool = True,
    ) -> None:
        self.strategies = strategies
        self.starting_equity = starting_equity
        self.risk_config = risk_config
        self.use_regime_detection = use_regime_detection

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
        backtest_risk_config = dict(self.risk_config or {})
        # Disable cooldown in backtests — wall-clock time doesn't advance
        backtest_risk_config.setdefault("cooldown_after_loss", 0)
        risk_mgr = RiskManager(
            account_equity=self.starting_equity,
            risk_config=backtest_risk_config,
        )
        regime_detector = RegimeDetector() if self.use_regime_detection else None
        signal_gen = SignalGenerator(
            strategies=self.strategies,
            risk_manager=risk_mgr,
            sqlite_engine=None,
            regime_detector=regime_detector,
        )
        executor = PaperExecutor(
            paper_mode=True,
            slippage_ticks_mean=1.5,
            slippage_ticks_std=0.5,
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

        # Pending signals from previous bar — filled at next bar's open
        pending_risk_results: list = []

        for bar in bars:
            # Phase 1: Fill pending signals from previous bar at this bar's open
            if pending_risk_results:
                for rr in pending_risk_results:
                    if rr.signal is not None:
                        rr.signal.entry_price = bar.open
                order_mgr.process_signals(pending_risk_results)
                pending_risk_results = []

            # Phase 2: Check open positions for exits on this bar
            closed_trades = order_mgr.on_bar(bar)
            results.trades.extend(closed_trades)

            # Phase 3: Compute indicators + regime
            snapshot = indicator_calc.update(bar)
            if regime_detector is not None:
                regime_detector.on_1m_bar(bar)

            # Phase 4: Generate and evaluate signals (fill on NEXT bar)
            risk_results = signal_gen.on_bar(bar, snapshot)
            results.signals_generated += len(risk_results)
            results.signals_rejected += sum(
                1 for r in risk_results if r.decision == RiskDecision.REJECTED
            )
            # Queue approved signals for next-bar fill
            pending_risk_results = [
                r for r in risk_results if r.decision != RiskDecision.REJECTED
            ]

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
