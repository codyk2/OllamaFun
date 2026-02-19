"""Domain models for the trading system."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Bar(BaseModel):
    """OHLCV bar data."""

    timestamp: datetime
    symbol: str = "MES"
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None


class IndicatorSnapshot(BaseModel):
    """Pre-computed indicator values at a point in time."""

    timestamp: datetime
    symbol: str = "MES"
    timeframe: str = "1m"
    vwap: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    keltner_upper: float | None = None
    keltner_middle: float | None = None
    keltner_lower: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    ema_9: float | None = None
    ema_21: float | None = None
    volume_profile_poc: float | None = None


class Signal(BaseModel):
    """Trading signal emitted by a strategy."""

    strategy: str
    symbol: str = "MES"
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: float
    stop_loss: float
    take_profit: float | None = None
    take_profit_primary: float | None = None
    take_profit_secondary: float | None = None
    reason: str = ""
    market_context: IndicatorSnapshot | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RiskResult(BaseModel):
    """Result of risk manager evaluation."""

    decision: RiskDecision
    position_size: int = 0
    reason: str = ""
    signal: Signal | None = None


class Trade(BaseModel):
    """A trade from entry to exit."""

    id: int | None = None
    strategy: str
    symbol: str = "MES"
    direction: Direction
    entry_price: float
    exit_price: float | None = None
    stop_loss: float
    take_profit: float | None = None
    quantity: int = 1
    entry_time: datetime
    exit_time: datetime | None = None
    status: TradeStatus = TradeStatus.OPEN
    pnl_ticks: float | None = None
    pnl_dollars: float | None = None
    risk_reward_actual: float | None = None
    commission: float = 1.24
    slippage_ticks: float = 0.0
    signal_confidence: float | None = None
    ai_review: str | None = None
    notes: str | None = None

    def calculate_pnl(self, tick_value: float = 1.25) -> None:
        """Calculate P&L when trade is closed."""
        if self.exit_price is None:
            return
        if self.direction == Direction.LONG:
            self.pnl_ticks = (self.exit_price - self.entry_price) / 0.25
        else:
            self.pnl_ticks = (self.entry_price - self.exit_price) / 0.25
        self.pnl_dollars = (self.pnl_ticks * tick_value * self.quantity) - self.commission

    def calculate_risk_reward(self) -> None:
        """Calculate actual risk:reward ratio."""
        if self.exit_price is None:
            return
        if self.direction == Direction.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.exit_price - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.exit_price
        if risk > 0:
            self.risk_reward_actual = reward / risk


class Position(BaseModel):
    """An open position being monitored."""

    trade: Trade
    current_price: float
    unrealized_pnl: float = 0.0
    trailing_stop: float | None = None
    trailing_activated: bool = False
    scale_out_done: bool = False
    original_quantity: int | None = None

    def update_price(self, price: float, tick_value: float = 1.25) -> None:
        """Update position with latest price."""
        self.current_price = price
        if self.trade.direction == Direction.LONG:
            pnl_ticks = (price - self.trade.entry_price) / 0.25
        else:
            pnl_ticks = (self.trade.entry_price - price) / 0.25
        self.unrealized_pnl = pnl_ticks * tick_value * self.trade.quantity

    def should_stop_out(self) -> bool:
        """Check if current price has hit stop loss."""
        stop = self.trailing_stop or self.trade.stop_loss
        if self.trade.direction == Direction.LONG:
            return self.current_price <= stop
        return self.current_price >= stop

    def should_take_profit(self) -> bool:
        """Check if current price has hit take profit."""
        if self.trade.take_profit is None:
            return False
        if self.trade.direction == Direction.LONG:
            return self.current_price >= self.trade.take_profit
        return self.current_price <= self.trade.take_profit


class RiskEvent(BaseModel):
    """A risk rule trigger event."""

    event_type: str
    details: dict
    severity: Severity
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EquitySnapshot(BaseModel):
    """Account equity at a point in time."""

    equity: float
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
