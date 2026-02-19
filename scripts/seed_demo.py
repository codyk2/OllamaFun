"""Seed the database with demo data for dashboard testing.

Generates sample bars, indicators, signals, and trades.

Usage: python -m scripts.seed_demo
"""

import random
from datetime import UTC, datetime, timedelta

from src.core.database import get_duckdb_connection, get_sqlite_engine, init_duckdb, init_sqlite_db
from src.core.logging import get_logger, setup_logging
from src.core.models import Direction, Signal, Trade, TradeStatus
from src.indicators.calculator import IndicatorCalculator
from src.journal.recorder import TradeRecorder
from src.market_data.historical import generate_sample_bars, store_bars_to_duckdb
from src.signals.generator import SignalGenerator
from src.strategies.mean_reversion import MeanReversionStrategy

logger = get_logger("seed_demo")


def main():
    setup_logging()
    logger.info("seeding_demo_data")

    # Init DBs
    engine = get_sqlite_engine()
    init_sqlite_db(engine)
    conn = get_duckdb_connection()
    init_duckdb(conn)

    # Generate and store sample bars
    bars = generate_sample_bars(count=500, start_price=5950.0, volatility=2.5)
    stored = store_bars_to_duckdb(conn, bars, timeframe="1m")
    print(f"Stored {stored} sample 1m bars")

    # Compute and store indicators
    calc = IndicatorCalculator()
    indicator_count = 0
    for bar in bars:
        snapshot = calc.update(bar)
        if snapshot:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO indicator_cache
                       (timestamp, symbol, timeframe, vwap, bb_upper, bb_middle, bb_lower,
                        keltner_upper, keltner_middle, keltner_lower, rsi_14, atr_14,
                        ema_9, ema_21, volume_profile_poc)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        snapshot.timestamp, snapshot.symbol, snapshot.timeframe,
                        snapshot.vwap, snapshot.bb_upper, snapshot.bb_middle, snapshot.bb_lower,
                        snapshot.keltner_upper, snapshot.keltner_middle, snapshot.keltner_lower,
                        snapshot.rsi_14, snapshot.atr_14, snapshot.ema_9, snapshot.ema_21,
                        snapshot.volume_profile_poc,
                    ],
                )
                indicator_count += 1
            except Exception as e:
                print(f"Error storing indicator: {e}")

    print(f"Stored {indicator_count} indicator snapshots")

    # Seed sample trades
    recorder = TradeRecorder(sqlite_engine=engine)
    random.seed(42)
    base_time = datetime(2024, 6, 15, 9, 30, tzinfo=UTC)
    trade_count = 0

    for i in range(15):
        direction = random.choice([Direction.LONG, Direction.SHORT])
        entry_price = 5950.0 + random.gauss(0, 10)
        pnl = random.gauss(5, 25)  # Slightly positive expectancy

        if direction == Direction.LONG:
            stop_loss = entry_price - random.uniform(2, 6)
            take_profit = entry_price + random.uniform(4, 12)
            exit_price = entry_price + (pnl / 1.25) * 0.25
        else:
            stop_loss = entry_price + random.uniform(2, 6)
            take_profit = entry_price - random.uniform(4, 12)
            exit_price = entry_price - (pnl / 1.25) * 0.25

        entry_time = base_time + timedelta(minutes=i * 30)
        exit_time = entry_time + timedelta(minutes=random.randint(5, 25))

        trade = Trade(
            strategy="mean_reversion",
            direction=direction,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            quantity=1,
            entry_time=entry_time,
            exit_time=exit_time,
            status=TradeStatus.CLOSED,
            pnl_dollars=round(pnl, 2),
            pnl_ticks=round(pnl / 1.25, 2),
            signal_confidence=round(random.uniform(0.5, 0.9), 2),
        )
        trade.calculate_risk_reward()
        recorder.record_trade(trade)
        trade_count += 1

    # Generate daily summary
    recorder.generate_daily_summary("2024-06-15")

    print(f"Stored {trade_count} sample trades")

    # Verify
    row_count = conn.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    ind_count = conn.execute("SELECT COUNT(*) FROM indicator_cache").fetchone()[0]
    print(f"DuckDB: {row_count} bars, {ind_count} indicators")

    conn.close()
    engine.dispose()
    print("Demo data seeded successfully!")


if __name__ == "__main__":
    main()
