"""Tests for OllamaClient."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.core.models import Direction, Trade, TradeStatus
from src.journal.analyzer import PerformanceMetrics
from src.llm.client import OllamaClient


def _make_trade(pnl: float = 50.0) -> Trade:
    return Trade(
        strategy="mean_reversion",
        direction=Direction.LONG,
        entry_price=5000.0,
        exit_price=5010.0,
        stop_loss=4996.0,
        take_profit=5008.0,
        quantity=1,
        entry_time=datetime.now(UTC),
        exit_time=datetime.now(UTC),
        status=TradeStatus.CLOSED,
        pnl_dollars=pnl,
        pnl_ticks=pnl / 1.25,
        signal_confidence=0.7,
    )


class TestPromptBuilding:
    def test_trade_review_prompt(self):
        client = OllamaClient(host="http://localhost:11434", model="test")
        trade = _make_trade(50.0)
        prompt = client._build_trade_review_prompt(trade)
        assert "LONG" in prompt
        assert "5000.00" in prompt
        assert "50.00" in prompt
        assert "mean_reversion" in prompt

    def test_daily_summary_prompt(self):
        client = OllamaClient(host="http://localhost:11434", model="test")
        trades = [_make_trade(50.0), _make_trade(-20.0)]
        metrics = PerformanceMetrics(
            total_trades=2,
            winners=1,
            losers=1,
            win_rate=0.5,
            net_pnl=30.0,
            profit_factor=2.5,
            max_drawdown=20.0,
            avg_winner=50.0,
            avg_loser=-20.0,
        )
        prompt = client._build_daily_summary_prompt(trades, metrics)
        assert "2" in prompt  # total trades
        assert "50.0%" in prompt  # win rate
        assert "$30.00" in prompt  # net pnl


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        client = OllamaClient(host="http://localhost:11434", model="test")
        mock_response = httpx.Response(
            200,
            json={"response": "This is a good trade."},
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.generate("test prompt")
            assert result == "This is a good trade."

    @pytest.mark.asyncio
    async def test_generate_with_system(self):
        client = OllamaClient(host="http://localhost:11434", model="test")
        mock_response = httpx.Response(
            200,
            json={"response": "Review complete."},
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.generate("test", system="You are a coach.")
            assert result == "Review complete."


class TestReviewTrade:
    @pytest.mark.asyncio
    async def test_review_trade(self):
        client = OllamaClient(host="http://localhost:11434", model="test")
        mock_response = httpx.Response(
            200,
            json={"response": "Good entry timing."},
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.review_trade(_make_trade())
            assert "Good entry timing." in result


class TestIsAvailable:
    @pytest.mark.asyncio
    async def test_available(self):
        client = OllamaClient(host="http://localhost:11434", model="test-model")
        mock_response = httpx.Response(
            200,
            json={"models": [{"name": "test-model:latest"}]},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            assert await client.is_available() is True

    @pytest.mark.asyncio
    async def test_not_available_no_model(self):
        client = OllamaClient(host="http://localhost:11434", model="missing-model")
        mock_response = httpx.Response(
            200,
            json={"models": [{"name": "other-model:latest"}]},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            assert await client.is_available() is False

    @pytest.mark.asyncio
    async def test_not_available_connection_error(self):
        client = OllamaClient(host="http://localhost:11434", model="test")

        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            assert await client.is_available() is False
