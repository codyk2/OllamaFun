"""Database setup for SQLite (trades/config) and DuckDB (market data)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    pass


# ── SQLite ORM Models ──────────────────────────────────────────────


class TradeRow(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String, nullable=False)
    symbol = Column(String, nullable=False, default="MES")
    direction = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float)
    quantity = Column(Integer, nullable=False, default=1)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    status = Column(String, nullable=False, default="OPEN")
    pnl_ticks = Column(Float)
    pnl_dollars = Column(Float)
    risk_reward_actual = Column(Float)
    commission = Column(Float, default=0.62)
    slippage_ticks = Column(Float, default=0)
    signal_confidence = Column(Float)
    ai_review = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class SignalRow(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String, nullable=False)
    symbol = Column(String, nullable=False, default="MES")
    direction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float)
    risk_approved = Column(Boolean, nullable=False)
    rejection_reason = Column(Text)
    executed = Column(Boolean, default=False)
    trade_id = Column(Integer)
    market_context = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class DailySummaryRow(Base):
    __tablename__ = "daily_summary"

    date = Column(String, primary_key=True)
    total_trades = Column(Integer, default=0)
    winners = Column(Integer, default=0)
    losers = Column(Integer, default=0)
    gross_pnl = Column(Float, default=0)
    net_pnl = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    win_rate = Column(Float, default=0)
    avg_winner = Column(Float, default=0)
    avg_loser = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    sharpe_daily = Column(Float)
    risk_events = Column(Integer, default=0)
    daily_limit_hit = Column(Boolean, default=False)
    ai_daily_review = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class RiskEventRow(Base):
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    details = Column(JSON, nullable=False)
    severity = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class StrategyConfigRow(Base):
    __tablename__ = "strategy_config"

    strategy_name = Column(String, primary_key=True)
    enabled = Column(Boolean, default=True)
    params = Column(JSON, nullable=False)
    last_modified = Column(DateTime, default=lambda: datetime.now(UTC))


class EquitySnapshotRow(Base):
    __tablename__ = "equity_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    equity = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0)
    realized_pnl_today = Column(Float, default=0)
    snapshot_time = Column(DateTime, default=lambda: datetime.now(UTC))


# ── SQLite Engine Setup ────────────────────────────────────────────


def _set_wal_mode(dbapi_conn, connection_record):
    """Enable WAL mode for SQLite on every connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def get_sqlite_engine(db_url: str | None = None):
    """Create SQLite engine with WAL mode enabled."""
    url = db_url or settings.db.sqlite_url
    # Ensure the directory exists
    db_path = url.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, echo=False)
    event.listen(engine, "connect", _set_wal_mode)
    return engine


def init_sqlite_db(engine=None) -> None:
    """Create all SQLite tables."""
    engine = engine or get_sqlite_engine()
    Base.metadata.create_all(engine)


def get_session(engine=None) -> Session:
    """Get a new SQLite session."""
    engine = engine or get_sqlite_engine()
    factory = sessionmaker(bind=engine)
    return factory()


# ── DuckDB Setup ───────────────────────────────────────────────────


def get_duckdb_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection for market data."""
    path = db_path or settings.db.duckdb_full_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_duckdb(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create DuckDB market data tables."""
    conn = conn or get_duckdb_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars_1m (
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL DEFAULT 'MES',
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            vwap DOUBLE,
            trade_count INTEGER,
            PRIMARY KEY (timestamp, symbol)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars_5m (
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL DEFAULT 'MES',
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            vwap DOUBLE,
            PRIMARY KEY (timestamp, symbol)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS indicator_cache (
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL DEFAULT 'MES',
            timeframe VARCHAR NOT NULL DEFAULT '1m',
            vwap DOUBLE,
            bb_upper DOUBLE,
            bb_middle DOUBLE,
            bb_lower DOUBLE,
            keltner_upper DOUBLE,
            keltner_middle DOUBLE,
            keltner_lower DOUBLE,
            rsi_14 DOUBLE,
            atr_14 DOUBLE,
            ema_9 DOUBLE,
            ema_21 DOUBLE,
            volume_profile_poc DOUBLE,
            PRIMARY KEY (timestamp, symbol, timeframe)
        )
    """)
