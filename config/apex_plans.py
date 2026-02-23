"""Apex Trader Funding plan presets.

Reference for all available Apex evaluation plans.
Use these to quickly configure accounts in accounts.json.

All plans: No daily drawdown, no scaling restrictions.
Tradovate plans include: TradingView, browser, mobile, Mac compatibility.
"""

APEX_PLANS = {
    "25K_FULL": {
        "equity": 25000.0,
        "max_contracts": 4,       # 40 micros
        "profit_goal": 1500.0,
        "trailing_drawdown": 1500.0,
        "monthly_cost": 187,
    },
    "50K_FULL": {
        "equity": 50000.0,
        "max_contracts": 10,      # 100 micros
        "profit_goal": 3000.0,
        "trailing_drawdown": 2500.0,
        "monthly_cost": 197,
    },
    "100K_FULL": {
        "equity": 100000.0,
        "max_contracts": 14,      # 140 micros
        "profit_goal": 6000.0,
        "trailing_drawdown": 3000.0,
        "monthly_cost": 297,
    },
    "150K_FULL": {
        "equity": 150000.0,
        "max_contracts": 17,      # 170 micros
        "profit_goal": 9000.0,
        "trailing_drawdown": 5000.0,
        "monthly_cost": 397,
    },
    "250K_FULL": {
        "equity": 250000.0,
        "max_contracts": 27,      # 270 micros
        "profit_goal": 15000.0,
        "trailing_drawdown": 6500.0,
        "monthly_cost": 397,
    },
    "300K_FULL": {
        "equity": 300000.0,
        "max_contracts": 35,      # 350 micros
        "profit_goal": 20000.0,
        "trailing_drawdown": 7500.0,
        "monthly_cost": 397,
    },
    "100K_STATIC": {
        "equity": 100000.0,
        "max_contracts": 2,       # 20 micros
        "profit_goal": 2000.0,
        "trailing_drawdown": 625.0,  # Static, not trailing
        "monthly_cost": 297,
    },
}

# Apex fees
EVAL_RESET_FEE_TRADOVATE = 100   # $100/reset for Tradovate plans
EVAL_RESET_FEE_RITHMIC = 80      # $80/reset for Rithmic plans
PA_MONTHLY_FEE = 85              # $85/month per funded (PA) account
MAX_PA_ACCOUNTS = 20


def get_plan(plan_name: str) -> dict:
    """Get an Apex plan by name. Raises KeyError if not found."""
    return APEX_PLANS[plan_name]


def make_account_config(
    plan_name: str,
    account_id: str,
    account_name: str | None = None,
    account_type: str = "EVAL",
) -> dict:
    """Generate an accounts.json entry from an Apex plan preset.

    Usage:
        config = make_account_config("50K_FULL", "apex-50k-1")
    """
    plan = APEX_PLANS[plan_name]
    return {
        "account_id": account_id,
        "name": account_name or f"Apex {plan_name.replace('_', ' ')}",
        "account_type": account_type,
        "tradovate_account_id": None,
        "equity": plan["equity"],
        "max_contracts": plan["max_contracts"],
        "trailing_drawdown": plan["trailing_drawdown"],
        "max_equity_high": plan["equity"],
        "profit_goal": plan["profit_goal"],
        "profit_split": 1.0,
        "enabled": True,
    }
