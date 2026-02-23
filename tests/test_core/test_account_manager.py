"""Tests for AccountManager — multi-account loading, filtering, and equity tracking."""

import json

import pytest

from src.core.account_manager import AccountManager
from src.core.models import Account, AccountType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_accounts_json(path, accounts: list[dict]) -> None:
    """Write a list of account dicts as JSON to *path*."""
    path.write_text(json.dumps(accounts), encoding="utf-8")


def _make_account_dict(
    account_id: str = "test-acct-1",
    name: str = "Test Account 1",
    account_type: str = "EVAL",
    equity: float = 50000.0,
    max_contracts: int = 10,
    trailing_drawdown: float = 3000.0,
    max_equity_high: float = 50000.0,
    profit_split: float = 1.0,
    enabled: bool = True,
    tradovate_account_id: int | None = None,
) -> dict:
    """Return a minimal account dict suitable for serialisation to JSON."""
    return {
        "account_id": account_id,
        "name": name,
        "account_type": account_type,
        "tradovate_account_id": tradovate_account_id,
        "equity": equity,
        "max_contracts": max_contracts,
        "trailing_drawdown": trailing_drawdown,
        "max_equity_high": max_equity_high,
        "profit_split": profit_split,
        "enabled": enabled,
    }


SINGLE_ENABLED = [_make_account_dict()]

TWO_ENABLED = [
    _make_account_dict(account_id="acct-1", name="Account 1"),
    _make_account_dict(account_id="acct-2", name="Account 2"),
]

MIXED_ENABLED_DISABLED = [
    _make_account_dict(account_id="enabled-1", name="Enabled 1", enabled=True),
    _make_account_dict(account_id="disabled-1", name="Disabled 1", enabled=False),
    _make_account_dict(account_id="enabled-2", name="Enabled 2", enabled=True),
    _make_account_dict(account_id="disabled-2", name="Disabled 2", enabled=False),
    _make_account_dict(account_id="disabled-3", name="Disabled 3", enabled=False),
]

ALL_DISABLED = [
    _make_account_dict(account_id="off-1", enabled=False),
    _make_account_dict(account_id="off-2", enabled=False),
]


# ---------------------------------------------------------------------------
# Tests: load_accounts
# ---------------------------------------------------------------------------

class TestLoadAccounts:
    """Tests for AccountManager.load_accounts()."""

    def test_load_single_account(self, tmp_path):
        """Loading a JSON file with one enabled account returns a list of one."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(cfg)
        result = mgr.load_accounts()

        assert len(result) == 1
        assert isinstance(result[0], Account)
        assert result[0].account_id == "test-acct-1"
        assert result[0].name == "Test Account 1"
        assert result[0].account_type == AccountType.EVAL
        assert result[0].equity == 50000.0
        assert result[0].enabled is True

    def test_load_multiple_accounts(self, tmp_path):
        """Loading a file with two accounts returns both."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, TWO_ENABLED)

        mgr = AccountManager(cfg)
        result = mgr.load_accounts()

        assert len(result) == 2
        ids = {a.account_id for a in result}
        assert ids == {"acct-1", "acct-2"}

    def test_load_returns_all_accounts_including_disabled(self, tmp_path):
        """load_accounts() returns *all* accounts, not just enabled ones."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, MIXED_ENABLED_DISABLED)

        mgr = AccountManager(cfg)
        result = mgr.load_accounts()

        assert len(result) == 5
        enabled_count = sum(1 for a in result if a.enabled)
        disabled_count = sum(1 for a in result if not a.enabled)
        assert enabled_count == 2
        assert disabled_count == 3

    def test_load_populates_internal_dict(self, tmp_path):
        """After loading, the internal accounts dict is keyed by account_id."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, TWO_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        assert "acct-1" in mgr.accounts
        assert "acct-2" in mgr.accounts
        assert len(mgr.accounts) == 2

    def test_load_clears_previous_accounts(self, tmp_path):
        """Calling load_accounts() twice replaces the old data."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, TWO_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()
        assert len(mgr.accounts) == 2

        # Overwrite with a single account and reload
        _write_accounts_json(cfg, SINGLE_ENABLED)
        mgr.load_accounts()
        assert len(mgr.accounts) == 1
        assert "test-acct-1" in mgr.accounts

    def test_load_preserves_all_fields(self, tmp_path):
        """All Account fields survive the JSON round-trip."""
        data = [_make_account_dict(
            account_id="full-field",
            name="Full Field Test",
            account_type="FUNDED",
            equity=75000.0,
            max_contracts=5,
            trailing_drawdown=2500.0,
            max_equity_high=76000.0,
            profit_split=0.9,
            enabled=True,
            tradovate_account_id=12345,
        )]
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, data)

        mgr = AccountManager(cfg)
        result = mgr.load_accounts()
        acct = result[0]

        assert acct.account_id == "full-field"
        assert acct.name == "Full Field Test"
        assert acct.account_type == AccountType.FUNDED
        assert acct.equity == 75000.0
        assert acct.max_contracts == 5
        assert acct.trailing_drawdown == 2500.0
        assert acct.max_equity_high == 76000.0
        assert acct.profit_split == 0.9
        assert acct.enabled is True
        assert acct.tradovate_account_id == 12345

    def test_load_empty_array_succeeds(self, tmp_path):
        """An empty JSON array is valid and results in zero accounts."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, [])

        mgr = AccountManager(cfg)
        result = mgr.load_accounts()

        assert result == []
        assert len(mgr.accounts) == 0


# ---------------------------------------------------------------------------
# Tests: load_accounts error paths
# ---------------------------------------------------------------------------

class TestLoadAccountsErrors:
    """Tests for error handling in load_accounts()."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Pointing at a nonexistent file raises FileNotFoundError."""
        cfg = tmp_path / "does_not_exist.json"
        mgr = AccountManager(cfg)

        with pytest.raises(FileNotFoundError):
            mgr.load_accounts()

    def test_invalid_json_raises_decode_error(self, tmp_path):
        """Malformed JSON raises json.JSONDecodeError."""
        cfg = tmp_path / "bad.json"
        cfg.write_text("{not valid json!!!", encoding="utf-8")

        mgr = AccountManager(cfg)

        with pytest.raises(json.JSONDecodeError):
            mgr.load_accounts()

    def test_json_object_instead_of_array_raises_value_error(self, tmp_path):
        """A JSON object (dict) instead of an array raises ValueError."""
        cfg = tmp_path / "obj.json"
        cfg.write_text(json.dumps({"account_id": "bad"}), encoding="utf-8")

        mgr = AccountManager(cfg)

        with pytest.raises(ValueError, match="JSON array"):
            mgr.load_accounts()

    def test_json_string_instead_of_array_raises_value_error(self, tmp_path):
        """A plain JSON string raises ValueError."""
        cfg = tmp_path / "str.json"
        cfg.write_text(json.dumps("just a string"), encoding="utf-8")

        mgr = AccountManager(cfg)

        with pytest.raises(ValueError, match="JSON array"):
            mgr.load_accounts()


# ---------------------------------------------------------------------------
# Tests: get_enabled_accounts
# ---------------------------------------------------------------------------

class TestGetEnabledAccounts:
    """Tests for AccountManager.get_enabled_accounts()."""

    def test_returns_only_enabled(self, tmp_path):
        """Only accounts with enabled=True are returned."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, MIXED_ENABLED_DISABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()
        enabled = mgr.get_enabled_accounts()

        assert len(enabled) == 2
        ids = {a.account_id for a in enabled}
        assert ids == {"enabled-1", "enabled-2"}
        assert all(a.enabled for a in enabled)

    def test_all_disabled_returns_empty(self, tmp_path):
        """When every account is disabled, the list is empty."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, ALL_DISABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        assert mgr.get_enabled_accounts() == []

    def test_all_enabled_returns_all(self, tmp_path):
        """When every account is enabled, all are returned."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, TWO_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()
        enabled = mgr.get_enabled_accounts()

        assert len(enabled) == 2

    def test_no_accounts_loaded_returns_empty(self, tmp_path):
        """Before loading, get_enabled_accounts returns empty list."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, MIXED_ENABLED_DISABLED)

        mgr = AccountManager(cfg)
        # Deliberately do NOT call load_accounts
        assert mgr.get_enabled_accounts() == []


# ---------------------------------------------------------------------------
# Tests: get_account
# ---------------------------------------------------------------------------

class TestGetAccount:
    """Tests for AccountManager.get_account()."""

    def test_returns_correct_account_by_id(self, tmp_path):
        """Exact ID lookup returns the right Account object."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, MIXED_ENABLED_DISABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        acct = mgr.get_account("enabled-1")
        assert acct is not None
        assert acct.account_id == "enabled-1"
        assert acct.name == "Enabled 1"

    def test_returns_disabled_account_by_id(self, tmp_path):
        """get_account() returns disabled accounts too — it does not filter."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, MIXED_ENABLED_DISABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        acct = mgr.get_account("disabled-1")
        assert acct is not None
        assert acct.enabled is False

    def test_returns_none_for_unknown_id(self, tmp_path):
        """An ID that was never loaded returns None."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        assert mgr.get_account("nonexistent-id") is None

    def test_returns_none_before_loading(self, tmp_path):
        """Before load_accounts() is called, every lookup returns None."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(cfg)
        assert mgr.get_account("test-acct-1") is None


# ---------------------------------------------------------------------------
# Tests: update_equity
# ---------------------------------------------------------------------------

class TestUpdateEquity:
    """Tests for AccountManager.update_equity()."""

    def test_updates_equity_value(self, tmp_path):
        """Basic equity update changes the account's equity field."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()
        mgr.update_equity("test-acct-1", 52000.0)

        acct = mgr.get_account("test-acct-1")
        assert acct.equity == 52000.0

    def test_updates_high_water_mark_on_new_high(self, tmp_path):
        """Equity above max_equity_high updates the high water mark."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        # Initial max_equity_high is 50000.0
        mgr.update_equity("test-acct-1", 53000.0)
        acct = mgr.get_account("test-acct-1")
        assert acct.max_equity_high == 53000.0

    def test_does_not_lower_high_water_mark(self, tmp_path):
        """Equity below max_equity_high leaves the high water mark unchanged."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, [_make_account_dict(
            account_id="hwm-test",
            equity=55000.0,
            max_equity_high=55000.0,
        )])

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        # Drawdown: equity drops but high water mark stays
        mgr.update_equity("hwm-test", 52000.0)
        acct = mgr.get_account("hwm-test")
        assert acct.equity == 52000.0
        assert acct.max_equity_high == 55000.0

    def test_successive_updates_track_highest(self, tmp_path):
        """A sequence of equity updates correctly tracks the running high."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        equities = [50500.0, 51000.0, 50800.0, 52000.0, 51500.0]
        for eq in equities:
            mgr.update_equity("test-acct-1", eq)

        acct = mgr.get_account("test-acct-1")
        assert acct.equity == 51500.0        # last value
        assert acct.max_equity_high == 52000.0  # peak value

    def test_unknown_account_raises_key_error(self, tmp_path):
        """Updating an account that does not exist raises KeyError."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        with pytest.raises(KeyError, match="account not found"):
            mgr.update_equity("ghost-account", 99999.0)

    def test_update_specific_account_in_multi(self, tmp_path):
        """With multiple accounts, update_equity targets only the specified one."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, TWO_ENABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        mgr.update_equity("acct-1", 60000.0)

        assert mgr.get_account("acct-1").equity == 60000.0
        assert mgr.get_account("acct-2").equity == 50000.0  # unchanged

    def test_update_equity_equal_to_high_water_mark(self, tmp_path):
        """Setting equity exactly equal to max_equity_high should not change it."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, [_make_account_dict(
            account_id="exact-hwm",
            equity=50000.0,
            max_equity_high=50000.0,
        )])

        mgr = AccountManager(cfg)
        mgr.load_accounts()
        mgr.update_equity("exact-hwm", 50000.0)

        acct = mgr.get_account("exact-hwm")
        assert acct.equity == 50000.0
        assert acct.max_equity_high == 50000.0


# ---------------------------------------------------------------------------
# Tests: mixed enabled/disabled integration
# ---------------------------------------------------------------------------

class TestMixedAccountScenarios:
    """Integration-style tests with multiple accounts in various states."""

    def test_mixed_load_then_filter(self, tmp_path):
        """Load 5 accounts (2 enabled, 3 disabled) and filter correctly."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, MIXED_ENABLED_DISABLED)

        mgr = AccountManager(cfg)
        all_accounts = mgr.load_accounts()
        enabled = mgr.get_enabled_accounts()

        assert len(all_accounts) == 5
        assert len(enabled) == 2
        for acct in enabled:
            assert acct.enabled is True

    def test_update_equity_on_disabled_account(self, tmp_path):
        """Equity can be updated on disabled accounts (they still exist)."""
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, MIXED_ENABLED_DISABLED)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        mgr.update_equity("disabled-1", 48000.0)
        acct = mgr.get_account("disabled-1")
        assert acct.equity == 48000.0
        assert acct.enabled is False

    def test_all_account_types_supported(self, tmp_path):
        """EVAL, FUNDED, and PAPER account types load correctly."""
        data = [
            _make_account_dict(account_id="eval-1", account_type="EVAL"),
            _make_account_dict(account_id="funded-1", account_type="FUNDED"),
            _make_account_dict(account_id="paper-1", account_type="PAPER"),
        ]
        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, data)

        mgr = AccountManager(cfg)
        mgr.load_accounts()

        assert mgr.get_account("eval-1").account_type == AccountType.EVAL
        assert mgr.get_account("funded-1").account_type == AccountType.FUNDED
        assert mgr.get_account("paper-1").account_type == AccountType.PAPER

    def test_config_path_stored_as_pathlib_path(self, tmp_path):
        """The config_path is always stored as a Path, even if passed as str."""
        from pathlib import Path

        cfg = tmp_path / "accounts.json"
        _write_accounts_json(cfg, SINGLE_ENABLED)

        mgr = AccountManager(str(cfg))
        assert isinstance(mgr.config_path, Path)
