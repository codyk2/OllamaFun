"""Bar aggregation: 5-second bars -> 1m/5m/15m OHLCV bars.

Collects incoming 5-second bars from IB and aggregates them into
standard timeframe bars, storing results in DuckDB.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import duckdb

from src.core.logging import get_logger
from src.core.models import Bar

logger = get_logger("aggregator")


class BarAggregator:
    """Aggregates 5-second bars into 1m and 5m OHLCV bars."""

    def __init__(
        self,
        duckdb_conn: duckdb.DuckDBPyConnection,
        on_1m_bar: Callable[[Bar], None] | None = None,
        on_5m_bar: Callable[[Bar], None] | None = None,
    ) -> None:
        self.conn = duckdb_conn
        self.on_1m_bar = on_1m_bar
        self.on_5m_bar = on_5m_bar

        # Working state for 1m bar accumulation
        self._current_1m: _BarBuilder | None = None
        # Accumulate 1m bars for 5m aggregation
        self._pending_1m_bars: list[Bar] = []

    def on_bar(self, bar: Bar) -> None:
        """Process an incoming 5-second bar."""
        minute_start = bar.timestamp.replace(second=0, microsecond=0)

        if self._current_1m is None:
            self._current_1m = _BarBuilder(minute_start, bar.symbol)

        # Check if we've crossed into a new minute
        if minute_start > self._current_1m.period_start:
            completed = self._current_1m.build()
            if completed:
                self._emit_1m_bar(completed)
            self._current_1m = _BarBuilder(minute_start, bar.symbol)

        self._current_1m.update(bar)

    def flush(self) -> None:
        """Flush any pending partial bar (e.g., at session end)."""
        if self._current_1m is not None:
            completed = self._current_1m.build()
            if completed:
                self._emit_1m_bar(completed)
            self._current_1m = None

    def _emit_1m_bar(self, bar: Bar) -> None:
        """Handle a completed 1-minute bar."""
        self._store_1m_bar(bar)

        if self.on_1m_bar:
            self.on_1m_bar(bar)

        # Accumulate for 5m aggregation
        self._pending_1m_bars.append(bar)
        if len(self._pending_1m_bars) >= 5:
            self._try_emit_5m_bar()

    def _try_emit_5m_bar(self) -> None:
        """Check if we have 5 consecutive 1m bars for a 5m bar."""
        if len(self._pending_1m_bars) < 5:
            return

        first = self._pending_1m_bars[0]
        five_min_start = first.timestamp.replace(
            minute=(first.timestamp.minute // 5) * 5, second=0, microsecond=0
        )

        # Check if the 5th bar completes the 5-minute period
        last = self._pending_1m_bars[-1]
        expected_end = five_min_start + timedelta(minutes=5)
        last_minute = last.timestamp.replace(second=0, microsecond=0)

        if last_minute >= expected_end - timedelta(minutes=1):
            bar_5m = _aggregate_bars(self._pending_1m_bars[:5], five_min_start)
            if bar_5m:
                self._store_5m_bar(bar_5m)
                if self.on_5m_bar:
                    self.on_5m_bar(bar_5m)
            self._pending_1m_bars = self._pending_1m_bars[5:]

    def _store_1m_bar(self, bar: Bar) -> None:
        """Insert a 1-minute bar into DuckDB."""
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO bars_1m
                   (timestamp, symbol, open, high, low, close, volume, vwap, trade_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    bar.timestamp, bar.symbol, bar.open, bar.high,
                    bar.low, bar.close, bar.volume, bar.vwap, bar.trade_count,
                ],
            )
        except Exception as e:
            logger.error("duckdb_1m_insert_failed", error=str(e))

    def _store_5m_bar(self, bar: Bar) -> None:
        """Insert a 5-minute bar into DuckDB."""
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO bars_5m
                   (timestamp, symbol, open, high, low, close, volume, vwap)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    bar.timestamp, bar.symbol, bar.open, bar.high,
                    bar.low, bar.close, bar.volume, bar.vwap,
                ],
            )
        except Exception as e:
            logger.error("duckdb_5m_insert_failed", error=str(e))


class _BarBuilder:
    """Accumulates ticks/sub-bars into a single OHLCV bar."""

    def __init__(self, period_start: datetime, symbol: str = "MES") -> None:
        self.period_start = period_start
        self.symbol = symbol
        self.open: float | None = None
        self.high: float = float("-inf")
        self.low: float = float("inf")
        self.close: float = 0.0
        self.volume: int = 0
        self.count: int = 0

    def update(self, bar: Bar) -> None:
        """Add a sub-bar's data."""
        if self.open is None:
            self.open = bar.open
        self.high = max(self.high, bar.high)
        self.low = min(self.low, bar.low)
        self.close = bar.close
        self.volume += bar.volume
        self.count += 1

    def build(self) -> Bar | None:
        """Build the completed bar. Returns None if no data."""
        if self.open is None or self.count == 0:
            return None
        return Bar(
            timestamp=self.period_start,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


def _aggregate_bars(bars: list[Bar], period_start: datetime) -> Bar | None:
    """Aggregate multiple bars into one."""
    if not bars:
        return None
    return Bar(
        timestamp=period_start,
        symbol=bars[0].symbol,
        open=bars[0].open,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=bars[-1].close,
        volume=sum(b.volume for b in bars),
    )
