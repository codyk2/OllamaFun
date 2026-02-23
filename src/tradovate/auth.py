"""Tradovate authentication with token caching and auto-refresh.

Handles OAuth-style token acquisition via the Tradovate REST API.
Tokens are cached for 1h50m (server issues 2h tokens) and refreshed
automatically when expired.
"""

from __future__ import annotations

import time

import httpx

from src.core.logging import get_logger

logger = get_logger("tradovate_auth")

# Tradovate access tokens are valid for 2 hours.
# We refresh 10 minutes early to avoid mid-request expiry.
_TOKEN_TTL_SECONDS: int = 110 * 60  # 1 hour 50 minutes


class TradovateAuth:
    """Manages Tradovate API authentication tokens.

    Acquires a token via POST /auth/accessTokenRequest and caches it.
    Automatically refreshes when the cached token is older than 1h50m.

    Parameters
    ----------
    config : object
        Must expose: base_url, username, password, app_id, app_version, cid, sec
    """

    def __init__(self, config: object) -> None:
        self._config = config
        self._client = httpx.AsyncClient(timeout=30.0)
        self._token: str | None = None
        self._token_expiry: float = 0.0

    async def get_token(self) -> str:
        """Return a valid access token, refreshing if expired.

        Returns
        -------
        str
            A valid Tradovate bearer token.

        Raises
        ------
        RuntimeError
            If the token request fails or the response is malformed.
        """
        if self._token is not None and time.monotonic() < self._token_expiry:
            return self._token

        await self._refresh_token()
        return self._token  # type: ignore[return-value]

    async def _refresh_token(self) -> None:
        """Request a new access token from Tradovate."""
        url = f"{self._config.base_url}/auth/accessTokenRequest"
        payload = {
            "name": self._config.username,
            "password": self._config.password,
            "appId": self._config.app_id,
            "appVersion": self._config.app_version,
            "cid": self._config.cid,
            "sec": self._config.sec,
        }

        logger.info("token_refresh_start", url=url)

        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "token_refresh_http_error",
                status=exc.response.status_code,
                body=exc.response.text[:500],
            )
            raise RuntimeError(
                f"Tradovate auth failed: HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("token_refresh_network_error", error=str(exc))
            raise RuntimeError(f"Tradovate auth network error: {exc}") from exc

        data = resp.json()
        token = data.get("accessToken")
        if not token:
            error_text = data.get("errorText", "unknown error")
            logger.error("token_refresh_no_token", response=data)
            raise RuntimeError(f"Tradovate auth returned no token: {error_text}")

        self._token = token
        self._token_expiry = time.monotonic() + _TOKEN_TTL_SECONDS

        logger.info("token_refresh_success", expires_in_s=_TOKEN_TTL_SECONDS)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        logger.info("auth_client_closed")
