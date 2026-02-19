"""Ollama LLM client for trade analysis and daily reviews."""

from __future__ import annotations

import httpx

from src.config import settings
from src.core.logging import get_logger
from src.core.models import Trade
from src.journal.analyzer import PerformanceMetrics

logger = get_logger("llm.client")


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.host = host or settings.ollama.host
        self.model = model or settings.ollama.model
        self.timeout = timeout

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt to Ollama and return the response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    async def review_trade(self, trade: Trade) -> str:
        """Generate an AI review of a completed trade."""
        prompt = self._build_trade_review_prompt(trade)
        system = (
            "You are a professional day trading coach reviewing MES micro futures trades. "
            "Analyze what went right or wrong, and give specific, actionable feedback. "
            "Be concise (3-5 sentences)."
        )
        return await self.generate(prompt, system=system)

    async def generate_daily_summary(
        self, trades: list[Trade], metrics: PerformanceMetrics
    ) -> str:
        """Generate an AI daily trading summary."""
        prompt = self._build_daily_summary_prompt(trades, metrics)
        system = (
            "You are a professional trading coach reviewing a day's performance. "
            "Identify patterns, mistakes, and areas for improvement. "
            "Be direct and constructive. Keep it under 200 words."
        )
        return await self.generate(prompt, system=system)

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is loaded."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False

    def _build_trade_review_prompt(self, trade: Trade) -> str:
        parts = [
            f"Trade Review:",
            f"  Direction: {trade.direction.value}",
            f"  Entry: {trade.entry_price:.2f}",
            f"  Exit: {trade.exit_price:.2f}" if trade.exit_price else "  Exit: (open)",
            f"  Stop: {trade.stop_loss:.2f}",
            f"  Target: {trade.take_profit:.2f}" if trade.take_profit else "  Target: none",
            f"  P&L: ${trade.pnl_dollars:.2f}" if trade.pnl_dollars is not None else "  P&L: N/A",
            f"  R:R Actual: {trade.risk_reward_actual:.2f}" if trade.risk_reward_actual else "",
            f"  Quantity: {trade.quantity}",
            f"  Strategy: {trade.strategy}",
            f"  Confidence: {trade.signal_confidence:.2f}" if trade.signal_confidence else "",
        ]

        parts.append("\nWhat went right or wrong? What should I do differently next time?")
        return "\n".join(p for p in parts if p)

    def _build_daily_summary_prompt(
        self, trades: list[Trade], metrics: PerformanceMetrics
    ) -> str:
        parts = [
            f"Daily Trading Summary:",
            f"  Total Trades: {metrics.total_trades}",
            f"  Win Rate: {metrics.win_rate:.1%}",
            f"  Net P&L: ${metrics.net_pnl:.2f}",
            f"  Profit Factor: {metrics.profit_factor:.2f}",
            f"  Max Drawdown: ${metrics.max_drawdown:.2f}",
            f"  Avg Winner: ${metrics.avg_winner:.2f}",
            f"  Avg Loser: ${metrics.avg_loser:.2f}",
            f"  Winning Streak: {metrics.winning_streak}",
            f"  Losing Streak: {metrics.losing_streak}",
            "",
            "Trades:",
        ]

        for i, t in enumerate(trades[:10], 1):
            pnl = f"${t.pnl_dollars:.2f}" if t.pnl_dollars is not None else "N/A"
            parts.append(f"  {i}. {t.direction.value} entry={t.entry_price:.2f} P&L={pnl}")

        parts.append("\nIdentify patterns, mistakes, and suggestions for improvement.")
        return "\n".join(parts)
