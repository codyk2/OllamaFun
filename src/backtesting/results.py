"""Backtest result container and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.models import Trade
from src.journal.analyzer import PerformanceMetrics, TradeAnalyzer


@dataclass
class BacktestResults:
    """Complete results of a backtest run."""

    strategy_name: str = "unknown"
    start_date: str = ""
    end_date: str = ""
    starting_equity: float = 10000.0
    ending_equity: float = 10000.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[float, float]] = field(default_factory=list)
    metrics: PerformanceMetrics | None = None
    bars_processed: int = 0
    signals_generated: int = 0
    signals_rejected: int = 0

    def compute_metrics(self) -> PerformanceMetrics:
        """Compute performance metrics from the trade list."""
        analyzer = TradeAnalyzer()
        self.metrics = analyzer.analyze(self.trades)
        self.equity_curve = analyzer.compute_equity_curve(
            self.trades, self.starting_equity
        )
        if self.equity_curve:
            self.ending_equity = self.equity_curve[-1][1]
        return self.metrics

    def summary(self) -> str:
        """Human-readable summary string."""
        if self.metrics is None:
            self.compute_metrics()
        m = self.metrics
        lines = [
            f"Backtest: {self.strategy_name}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Bars processed: {self.bars_processed}",
            f"Signals: {self.signals_generated} generated, {self.signals_rejected} rejected",
            f"Trades: {m.total_trades} ({m.winners}W / {m.losers}L / {m.breakeven}BE)",
            f"Win rate: {m.win_rate:.1%}",
            f"Net P&L: ${m.net_pnl:.2f}",
            f"Profit factor: {m.profit_factor:.2f}",
            f"Max drawdown: ${m.max_drawdown:.2f} ({m.max_drawdown_pct:.1%})",
            f"Sharpe: {m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "Sharpe: N/A",
            f"Equity: ${self.starting_equity:.2f} -> ${self.ending_equity:.2f}",
        ]
        return "\n".join(lines)
