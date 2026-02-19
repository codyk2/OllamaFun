"""Historical data loader for backtesting and dashboard seeding.

Supports loading from CSV files and from IB historical data requests.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from src.core.logging import get_logger
from src.core.models import Bar

logger = get_logger("historical")


def load_csv_bars(
    file_path: str | Path,
    symbol: str = "MES",
    timestamp_col: str = "timestamp",
    date_format: str | None = None,
) -> list[Bar]:
    """Load bars from a CSV file.

    Expected columns: timestamp, open, high, low, close, volume
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("csv_not_found", path=str(path))
        return []

    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        if date_format:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_format)
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        bars = []
        for _, row in df.iterrows():
            bars.append(Bar(
                timestamp=row[timestamp_col].to_pydatetime(),
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            ))

        logger.info("csv_loaded", path=str(path), bars=len(bars))
        return bars
    except Exception as e:
        logger.error("csv_load_failed", path=str(path), error=str(e))
        return []


def store_bars_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    bars: list[Bar],
    timeframe: str = "1m",
) -> int:
    """Store bars into DuckDB. Returns count of rows inserted."""
    if not bars:
        return 0

    table = f"bars_{timeframe}"
    data = [
        (b.timestamp, b.symbol, b.open, b.high, b.low, b.close, b.volume, b.vwap, b.trade_count)
        for b in bars
    ]

    try:
        # Use INSERT OR REPLACE for idempotency
        conn.executemany(
            f"""INSERT OR REPLACE INTO {table}
                (timestamp, symbol, open, high, low, close, volume, vwap, trade_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
        logger.info("bars_stored", table=table, count=len(data))
        return len(data)
    except Exception as e:
        logger.error("bars_store_failed", table=table, error=str(e))
        return 0


def generate_sample_bars(
    count: int = 500,
    start_price: float = 5000.0,
    symbol: str = "MES",
    volatility: float = 2.0,
) -> list[Bar]:
    """Generate synthetic MES bars for testing/demo purposes.

    Creates realistic-looking price action with random walks.
    """
    import random

    bars = []
    price = start_price
    base_time = datetime(2025, 1, 15, 17, 5)  # Wednesday session open

    for i in range(count):
        # Random walk with slight mean reversion
        change = random.gauss(0, volatility * 0.25)
        if abs(price - start_price) > 20:
            change -= (price - start_price) * 0.01  # Mean reversion pull

        open_price = price
        close_price = price + change

        # Generate realistic OHLC from the move
        if change >= 0:
            high_price = close_price + abs(random.gauss(0, volatility * 0.15))
            low_price = open_price - abs(random.gauss(0, volatility * 0.15))
        else:
            high_price = open_price + abs(random.gauss(0, volatility * 0.15))
            low_price = close_price - abs(random.gauss(0, volatility * 0.15))

        # Round to tick size
        open_price = round(round(open_price / 0.25) * 0.25, 2)
        high_price = round(round(high_price / 0.25) * 0.25, 2)
        low_price = round(round(low_price / 0.25) * 0.25, 2)
        close_price = round(round(close_price / 0.25) * 0.25, 2)

        # Ensure OHLC integrity
        high_price = max(open_price, close_price, high_price)
        low_price = min(open_price, close_price, low_price)

        volume = int(random.gauss(1500, 500))
        volume = max(volume, 100)

        timestamp = base_time + pd.Timedelta(minutes=i)

        bars.append(Bar(
            timestamp=timestamp,
            symbol=symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        ))

        price = close_price

    logger.info("sample_bars_generated", count=count, start=start_price)
    return bars
