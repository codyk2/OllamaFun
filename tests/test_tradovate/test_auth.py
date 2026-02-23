"""Tests for TradovateAuth token management."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.config import TradovateConfig
from src.tradovate.auth import TradovateAuth, _TOKEN_TTL_SECONDS


@pytest.fixture
def config():
    """Minimal Tradovate config for testing."""
    return TradovateConfig(
        TRADOVATE_USERNAME="test",
        TRADOVATE_PASSWORD="test",
        TRADOVATE_APP_ID="test",
        TRADOVATE_CID="1",
        TRADOVATE_SEC="secret",
        TRADOVATE_ENV="demo",
    )


@pytest.fixture
def auth(config):
    """TradovateAuth instance with a real config."""
    return TradovateAuth(config)


def _mock_token_response(token: str = "tok_abc123") -> httpx.Response:
    """Build a fake successful token response."""
    return httpx.Response(
        status_code=200,
        json={"accessToken": token},
        request=httpx.Request("POST", "https://demo.tradovateapi.com/v1/auth/accessTokenRequest"),
    )


def _mock_error_response(status: int = 401, body: str = "Unauthorized") -> httpx.Response:
    """Build a fake error response."""
    return httpx.Response(
        status_code=status,
        text=body,
        request=httpx.Request("POST", "https://demo.tradovateapi.com/v1/auth/accessTokenRequest"),
    )


def _mock_no_token_response() -> httpx.Response:
    """Build a response with no accessToken field."""
    return httpx.Response(
        status_code=200,
        json={"errorText": "Invalid credentials"},
        request=httpx.Request("POST", "https://demo.tradovateapi.com/v1/auth/accessTokenRequest"),
    )


class TestInitialState:
    """TradovateAuth starts with no cached token."""

    def test_token_is_none(self, auth):
        assert auth._token is None

    def test_token_expiry_is_zero(self, auth):
        assert auth._token_expiry == 0.0

    def test_client_is_created(self, auth):
        assert isinstance(auth._client, httpx.AsyncClient)


class TestGetToken:
    """get_token() acquires a token via HTTP and caches it."""

    async def test_makes_http_request_and_returns_token(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_token_response("tok_first")
        auth._client = mock_client

        token = await auth.get_token()

        assert token == "tok_first"
        mock_client.post.assert_called_once()

        # Verify the URL and payload
        call_args = mock_client.post.call_args
        assert "/auth/accessTokenRequest" in call_args.args[0]
        payload = call_args.kwargs["json"]
        assert payload["name"] == "test"
        assert payload["password"] == "test"
        assert payload["appId"] == "test"
        assert payload["cid"] == "1"
        assert payload["sec"] == "secret"

    async def test_caches_token_after_first_call(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_token_response("tok_cached")
        auth._client = mock_client

        await auth.get_token()

        assert auth._token == "tok_cached"
        assert auth._token_expiry > 0.0

    async def test_returns_cached_token_without_http_call(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_token_response("tok_cached")
        auth._client = mock_client

        first = await auth.get_token()
        second = await auth.get_token()

        assert first == second == "tok_cached"
        # Only one HTTP call should have been made
        assert mock_client.post.call_count == 1

    async def test_http_error_raises_runtime_error(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        resp = _mock_error_response(401, "Unauthorized")
        mock_client.post.return_value = resp
        auth._client = mock_client

        with pytest.raises(RuntimeError, match="Tradovate auth failed: HTTP 401"):
            await auth.get_token()

    async def test_network_error_raises_runtime_error(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        auth._client = mock_client

        with pytest.raises(RuntimeError, match="Tradovate auth network error"):
            await auth.get_token()

    async def test_missing_access_token_raises_runtime_error(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_no_token_response()
        auth._client = mock_client

        with pytest.raises(RuntimeError, match="Tradovate auth returned no token"):
            await auth.get_token()


class TestTokenExpiry:
    """get_token() refreshes when the cached token has expired."""

    async def test_refreshes_when_expired(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = [
            _mock_token_response("tok_old"),
            _mock_token_response("tok_new"),
        ]
        auth._client = mock_client

        # First call — acquires token
        first = await auth.get_token()
        assert first == "tok_old"
        assert mock_client.post.call_count == 1

        # Simulate expiry by rewinding the expiry timestamp
        auth._token_expiry = 0.0

        # Second call — should refresh
        second = await auth.get_token()
        assert second == "tok_new"
        assert mock_client.post.call_count == 2

    @patch("src.tradovate.auth.time")
    async def test_refreshes_based_on_monotonic_clock(self, mock_time, auth):
        """Token is considered expired when time.monotonic() >= _token_expiry."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = [
            _mock_token_response("tok_v1"),
            _mock_token_response("tok_v2"),
        ]
        auth._client = mock_client

        # Time starts at 1000
        mock_time.monotonic.return_value = 1000.0

        first = await auth.get_token()
        assert first == "tok_v1"
        # Expiry should be set to 1000 + _TOKEN_TTL_SECONDS
        assert auth._token_expiry == 1000.0 + _TOKEN_TTL_SECONDS

        # Advance time but still within TTL — should return cached
        mock_time.monotonic.return_value = 1000.0 + _TOKEN_TTL_SECONDS - 1
        second = await auth.get_token()
        assert second == "tok_v1"
        assert mock_client.post.call_count == 1

        # Advance time past TTL — should refresh
        mock_time.monotonic.return_value = 1000.0 + _TOKEN_TTL_SECONDS + 1
        third = await auth.get_token()
        assert third == "tok_v2"
        assert mock_client.post.call_count == 2

    async def test_token_ttl_is_110_minutes(self):
        """Sanity check that the TTL constant is 1h50m (110 * 60 seconds)."""
        assert _TOKEN_TTL_SECONDS == 110 * 60


class TestClose:
    """close() shuts down the HTTP client cleanly."""

    async def test_close_calls_aclose(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        auth._client = mock_client

        await auth.close()

        mock_client.aclose.assert_called_once()

    async def test_close_without_prior_usage(self, config):
        """close() works even if get_token() was never called."""
        a = TradovateAuth(config)
        # Should not raise
        await a.close()

    async def test_close_after_token_acquired(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_token_response("tok_x")
        auth._client = mock_client

        await auth.get_token()
        await auth.close()

        mock_client.aclose.assert_called_once()


class TestConfigIntegration:
    """Verify the config properties feed through correctly."""

    def test_demo_base_url(self, config):
        assert config.base_url == "https://demo.tradovateapi.com/v1"

    def test_live_base_url(self):
        live_config = TradovateConfig(
            TRADOVATE_USERNAME="test",
            TRADOVATE_PASSWORD="test",
            TRADOVATE_APP_ID="test",
            TRADOVATE_CID="1",
            TRADOVATE_SEC="secret",
            TRADOVATE_ENV="live",
        )
        assert live_config.base_url == "https://live.tradovateapi.com/v1"

    async def test_auth_uses_config_base_url(self, auth):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_token_response()
        auth._client = mock_client

        await auth.get_token()

        call_url = mock_client.post.call_args.args[0]
        assert call_url == "https://demo.tradovateapi.com/v1/auth/accessTokenRequest"
