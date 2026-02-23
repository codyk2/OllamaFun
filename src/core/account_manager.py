"""Multi-account management for prop firm copy trading.

Loads account configurations from JSON, tracks equity and high water marks,
and provides lookup for enabled accounts.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.logging import get_logger
from src.core.models import Account

logger = get_logger("account_manager")


class AccountManager:
    """Manages multiple trading accounts loaded from a JSON config file.

    Each account tracks its own equity, trailing drawdown high water mark,
    and enabled/disabled state.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.accounts: dict[str, Account] = {}

    def load_accounts(self) -> list[Account]:
        """Read the JSON config file, parse into Account models, and store them.

        Returns:
            List of all loaded Account objects.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the config file contains invalid JSON.
            ValueError: If the config file does not contain a JSON array.
        """
        logger.info("loading_accounts", path=str(self.config_path))

        raw_text = self.config_path.read_text(encoding="utf-8")
        raw_data = json.loads(raw_text)

        if not isinstance(raw_data, list):
            raise ValueError(
                f"accounts config must be a JSON array, got {type(raw_data).__name__}"
            )

        self.accounts.clear()
        loaded: list[Account] = []

        for entry in raw_data:
            account = Account(**entry)
            self.accounts[account.account_id] = account
            loaded.append(account)

        logger.info(
            "accounts_loaded",
            total=len(loaded),
            enabled=len([a for a in loaded if a.enabled]),
        )
        return loaded

    def get_enabled_accounts(self) -> list[Account]:
        """Return only accounts that are currently enabled.

        Returns:
            List of Account objects where ``enabled`` is True.
        """
        return [a for a in self.accounts.values() if a.enabled]

    def get_account(self, account_id: str) -> Account | None:
        """Look up an account by its ID.

        Args:
            account_id: The unique account identifier.

        Returns:
            The Account if found, otherwise None.
        """
        return self.accounts.get(account_id)

    def update_equity(self, account_id: str, equity: float) -> None:
        """Update an account's equity and track the high water mark.

        If the new equity exceeds the current ``max_equity_high``, the high
        water mark is updated. This is essential for trailing drawdown
        calculations used by prop firm evaluation rules.

        Args:
            account_id: The account to update.
            equity: The new equity value.

        Raises:
            KeyError: If the account_id is not found.
        """
        account = self.accounts.get(account_id)
        if account is None:
            raise KeyError(f"account not found: {account_id}")

        previous_equity = account.equity
        account.equity = equity

        if equity > account.max_equity_high:
            account.max_equity_high = equity
            logger.info(
                "new_equity_high",
                account=account_id,
                equity=equity,
                previous_high=account.max_equity_high,
            )

        logger.debug(
            "equity_updated",
            account=account_id,
            equity=equity,
            previous=previous_equity,
            high_water=account.max_equity_high,
        )
