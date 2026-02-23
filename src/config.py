"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent


class TradovateConfig(BaseSettings):
    username: str = Field(default="", alias="TRADOVATE_USERNAME")
    password: str = Field(default="", alias="TRADOVATE_PASSWORD")
    app_id: str = Field(default="", alias="TRADOVATE_APP_ID")
    app_version: str = Field(default="1.0.0", alias="TRADOVATE_APP_VERSION")
    cid: str = Field(default="", alias="TRADOVATE_CID")
    sec: str = Field(default="", alias="TRADOVATE_SEC")
    env: str = Field(default="demo", alias="TRADOVATE_ENV")

    @property
    def base_url(self) -> str:
        if self.env == "demo":
            return "https://demo.tradovateapi.com/v1"
        return "https://live.tradovateapi.com/v1"

    @property
    def md_ws_url(self) -> str:
        if self.env == "demo":
            return "wss://md-d.tradovateapi.com/v1/websocket"
        return "wss://md.tradovateapi.com/v1/websocket"

    @property
    def order_ws_url(self) -> str:
        if self.env == "demo":
            return "wss://demo.tradovateapi.com/v1/websocket"
        return "wss://live.tradovateapi.com/v1/websocket"


class TradingConfig(BaseSettings):
    symbol: str = Field(default="MES", alias="TRADING_SYMBOL")
    paper_mode: bool = Field(default=True, alias="PAPER_MODE")
    account_config_path: str = Field(
        default="config/accounts.json", alias="ACCOUNT_CONFIG_PATH"
    )


class OllamaConfig(BaseSettings):
    host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    model: str = Field(default="mistral:7b-instruct-v0.3-q4_K_M", alias="OLLAMA_MODEL")


class DatabaseConfig(BaseSettings):
    sqlite_path: str = Field(default="db/trading.db", alias="SQLITE_DB_PATH")
    duckdb_path: str = Field(default="db/market_data.duckdb", alias="DUCKDB_PATH")

    @property
    def sqlite_url(self) -> str:
        path = PROJECT_ROOT / self.sqlite_path
        return f"sqlite:///{path}"

    @property
    def duckdb_full_path(self) -> Path:
        return PROJECT_ROOT / self.duckdb_path


class LogConfig(BaseSettings):
    level: str = Field(default="INFO", alias="LOG_LEVEL")
    format: str = Field(default="json", alias="LOG_FORMAT")


class DashboardConfig(BaseSettings):
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")


# MES contract specification — reference data, not configurable
MES_SPEC = {
    "symbol": "MES",
    "exchange": "CME",
    "tick_size": 0.25,
    "tick_value": 1.25,
    "commission_per_side": 0.52,  # ~$1.04 round-trip (Tradovate/Apex fees)
    "currency": "USD",
}

# Risk defaults — user can tighten but NEVER loosen
RISK_DEFAULTS = {
    "max_risk_per_trade": 0.015,
    "daily_loss_limit": 0.03,
    "weekly_loss_limit": 0.06,
    "max_daily_trades": 10,
    "min_risk_reward_ratio": 0.8,
    "max_concurrent_positions": 2,
    "always_use_stop_loss": True,
    "max_stop_distance_atr": 2.0,
    "cooldown_after_loss": 60,
    "max_position_size": 2,
    "skip_first_minutes": 5,
    "skip_last_minutes": 5,
}


class Settings:
    """Aggregated application settings."""

    def __init__(self) -> None:
        self.tradovate = TradovateConfig()
        self.trading = TradingConfig()
        self.ollama = OllamaConfig()
        self.db = DatabaseConfig()
        self.log = LogConfig()
        self.dashboard = DashboardConfig()
        self.mes = MES_SPEC
        self.risk = RISK_DEFAULTS


settings = Settings()
