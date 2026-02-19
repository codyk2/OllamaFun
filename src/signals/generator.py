"""Signal generation pipeline.

Connects strategies to risk management. Every signal passes through
RiskManager before it can be executed.
"""

from __future__ import annotations

from src.core.database import SignalRow, get_session
from src.core.logging import get_logger
from src.core.models import Bar, IndicatorSnapshot, RiskDecision, RiskResult
from src.indicators.regime import RegimeDetector
from src.risk.manager import RiskManager
from src.signals.scorer import score_confluence
from src.strategies.base import BaseStrategy

logger = get_logger("signals")


class SignalGenerator:
    """Processes bars through strategies and risk checks.

    Pipeline: bar -> strategy.on_bar() -> scorer -> risk_manager.evaluate() -> persist
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        risk_manager: RiskManager,
        sqlite_engine=None,
        regime_detector: RegimeDetector | None = None,
    ) -> None:
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.sqlite_engine = sqlite_engine
        self.regime_detector = regime_detector

    def on_bar(
        self, bar: Bar, snapshot: IndicatorSnapshot | None
    ) -> list[RiskResult]:
        """Process a bar through all strategies.

        Returns list of RiskResults (both approved and rejected).
        """
        results: list[RiskResult] = []

        if snapshot is None:
            return results

        for strategy in self.strategies:
            signal = strategy.on_bar(bar, snapshot)
            if signal is None:
                continue

            # Post-hoc confluence scoring adjustment
            confluence = score_confluence(signal, snapshot)
            signal.confidence = (signal.confidence + confluence) / 2

            # Regime scaling: reduce confidence in non-ranging markets
            if self.regime_detector is not None:
                scaling = self.regime_detector.state.signal_scaling
                if scaling <= 0.0:
                    continue  # Skip signal entirely in trending regime
                signal.confidence *= scaling

            # Risk gate
            atr = snapshot.atr_14 or 3.0
            risk_result = self.risk_manager.evaluate(
                signal,
                atr=atr,
                current_time=bar.timestamp,
                trading_window=strategy.config.trading_window,
            )

            # Persist signal
            self._persist_signal(signal, risk_result)

            logger.info(
                "signal_processed",
                strategy=strategy.name,
                direction=signal.direction.value,
                confidence=f"{signal.confidence:.2f}",
                decision=risk_result.decision.value,
                reason=risk_result.reason,
            )

            results.append(risk_result)

        return results

    def _persist_signal(self, signal, result: RiskResult) -> None:
        """Store signal in SQLite for journal/analysis."""
        if self.sqlite_engine is None:
            return
        try:
            session = get_session(self.sqlite_engine)
            row = SignalRow(
                strategy=signal.strategy,
                symbol=signal.symbol,
                direction=signal.direction.value,
                confidence=signal.confidence,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_approved=(result.decision == RiskDecision.APPROVED),
                rejection_reason=(
                    result.reason if result.decision == RiskDecision.REJECTED else None
                ),
                executed=False,
                market_context=(
                    signal.market_context.model_dump() if signal.market_context else None
                ),
            )
            session.add(row)
            session.commit()
            session.close()
        except Exception as e:
            logger.error("signal_persist_failed", error=str(e))
