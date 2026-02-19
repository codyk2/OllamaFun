"""Trade performance analysis and metrics computation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.models import Trade, TradeStatus


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics for a set of trades."""

    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_winner: float = 0.0
    max_loser: float = 0.0
    avg_risk_reward: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float | None = None
    winning_streak: int = 0
    losing_streak: int = 0
    current_streak: int = 0
    avg_trade_duration_minutes: float = 0.0


class TradeAnalyzer:
    """Computes performance metrics from a list of trades."""

    def analyze(self, trades: list[Trade]) -> PerformanceMetrics:
        """Compute all metrics from completed trades."""
        closed = [t for t in trades if t.status == TradeStatus.CLOSED and t.pnl_dollars is not None]

        if not closed:
            return PerformanceMetrics()

        metrics = PerformanceMetrics(total_trades=len(closed))

        # Classify trades
        winners = [t for t in closed if t.pnl_dollars > 0]
        losers = [t for t in closed if t.pnl_dollars < 0]
        breakeven = [t for t in closed if t.pnl_dollars == 0]

        metrics.winners = len(winners)
        metrics.losers = len(losers)
        metrics.breakeven = len(breakeven)
        metrics.win_rate = metrics.winners / metrics.total_trades if metrics.total_trades > 0 else 0.0

        # P&L
        metrics.gross_profit = sum(t.pnl_dollars for t in winners)
        metrics.gross_loss = sum(t.pnl_dollars for t in losers)  # Negative number
        metrics.net_pnl = metrics.gross_profit + metrics.gross_loss

        # Averages
        metrics.avg_winner = metrics.gross_profit / len(winners) if winners else 0.0
        metrics.avg_loser = metrics.gross_loss / len(losers) if losers else 0.0
        metrics.max_winner = max((t.pnl_dollars for t in winners), default=0.0)
        metrics.max_loser = min((t.pnl_dollars for t in losers), default=0.0)

        # Profit factor
        if metrics.gross_loss != 0:
            metrics.profit_factor = abs(metrics.gross_profit / metrics.gross_loss)
        elif metrics.gross_profit > 0:
            metrics.profit_factor = float("inf")

        # Risk:reward
        rr_values = [t.risk_reward_actual for t in closed if t.risk_reward_actual is not None]
        metrics.avg_risk_reward = sum(rr_values) / len(rr_values) if rr_values else 0.0

        # Drawdown
        metrics.max_drawdown, metrics.max_drawdown_pct = self._compute_max_drawdown(closed)

        # Sharpe
        pnl_series = [t.pnl_dollars for t in closed]
        metrics.sharpe_ratio = self._compute_sharpe(pnl_series)

        # Streaks
        metrics.winning_streak, metrics.losing_streak, metrics.current_streak = (
            self._compute_streaks(closed)
        )

        # Duration
        durations = []
        for t in closed:
            if t.entry_time and t.exit_time:
                delta = (t.exit_time - t.entry_time).total_seconds() / 60.0
                durations.append(delta)
        metrics.avg_trade_duration_minutes = (
            sum(durations) / len(durations) if durations else 0.0
        )

        return metrics

    def compute_equity_curve(
        self, trades: list[Trade], starting_equity: float
    ) -> list[tuple[float, float]]:
        """Build equity curve: list of (timestamp_epoch, equity) tuples."""
        closed = [t for t in trades if t.status == TradeStatus.CLOSED and t.pnl_dollars is not None]
        closed.sort(key=lambda t: t.exit_time or t.entry_time)

        curve = []
        equity = starting_equity

        for trade in closed:
            equity += trade.pnl_dollars
            ts = (trade.exit_time or trade.entry_time).timestamp()
            curve.append((ts, equity))

        return curve

    def compute_drawdown_series(
        self, equity_curve: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Compute drawdown at each point in the equity curve."""
        if not equity_curve:
            return []

        peak = equity_curve[0][1]
        drawdowns = []

        for ts, equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            drawdowns.append((ts, dd))

        return drawdowns

    def _compute_max_drawdown(self, trades: list[Trade]) -> tuple[float, float]:
        """Compute max drawdown in dollars and percentage."""
        if not trades:
            return 0.0, 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        max_dd_pct = 0.0

        for trade in trades:
            cumulative += trade.pnl_dollars
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
                if peak > 0:
                    max_dd_pct = dd / peak

        return max_dd, max_dd_pct

    def _compute_sharpe(
        self, pnl_series: list[float], periods_per_year: float = 252.0
    ) -> float | None:
        """Annualized Sharpe ratio from trade P&L series."""
        if len(pnl_series) < 2:
            return None

        mean_pnl = sum(pnl_series) / len(pnl_series)
        variance = sum((p - mean_pnl) ** 2 for p in pnl_series) / (len(pnl_series) - 1)
        std_pnl = math.sqrt(variance)

        if std_pnl == 0:
            return None

        return (mean_pnl / std_pnl) * math.sqrt(periods_per_year)

    def _compute_streaks(self, trades: list[Trade]) -> tuple[int, int, int]:
        """Returns (max_winning_streak, max_losing_streak, current_streak).

        current_streak: positive = winning, negative = losing.
        """
        if not trades:
            return 0, 0, 0

        max_win_streak = 0
        max_lose_streak = 0
        current = 0

        for trade in trades:
            if trade.pnl_dollars > 0:
                if current > 0:
                    current += 1
                else:
                    current = 1
                max_win_streak = max(max_win_streak, current)
            elif trade.pnl_dollars < 0:
                if current < 0:
                    current -= 1
                else:
                    current = -1
                max_lose_streak = max(max_lose_streak, abs(current))
            else:
                current = 0

        return max_win_streak, max_lose_streak, current
