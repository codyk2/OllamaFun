"""Health check and system monitoring.

Periodically checks IB connection, database writes, and Ollama availability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.core.logging import get_logger

logger = get_logger("health")


class ServiceStatus(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthStatus:
    """Current health of all services."""

    ib_status: ServiceStatus = ServiceStatus.UNKNOWN
    ib_last_heartbeat: datetime | None = None
    duckdb_status: ServiceStatus = ServiceStatus.UNKNOWN
    sqlite_status: ServiceStatus = ServiceStatus.UNKNOWN
    ollama_status: ServiceStatus = ServiceStatus.UNKNOWN
    uptime_seconds: float = 0.0
    last_check: datetime | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def overall_status(self) -> ServiceStatus:
        """Overall system health."""
        statuses = [self.ib_status, self.duckdb_status, self.sqlite_status]
        if any(s == ServiceStatus.DOWN for s in statuses):
            return ServiceStatus.DOWN
        if any(s == ServiceStatus.DEGRADED for s in statuses):
            return ServiceStatus.DEGRADED
        if all(s == ServiceStatus.UP for s in statuses):
            return ServiceStatus.UP
        return ServiceStatus.UNKNOWN


class HealthMonitor:
    """Monitors system health and tracks service availability."""

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self.status = HealthStatus()
        self._ib_provider = None
        self._duckdb_conn = None
        self._sqlite_engine = None

    def register_ib(self, provider) -> None:
        """Register IB provider for health monitoring."""
        self._ib_provider = provider

    def register_duckdb(self, conn) -> None:
        """Register DuckDB connection."""
        self._duckdb_conn = conn

    def register_sqlite(self, engine) -> None:
        """Register SQLite engine."""
        self._sqlite_engine = engine

    def check_all(self) -> HealthStatus:
        """Run all health checks."""
        self.status.uptime_seconds = time.monotonic() - self._start_time
        self.status.last_check = datetime.now()
        self.status.errors.clear()

        self._check_ib()
        self._check_duckdb()
        self._check_sqlite()

        if self.status.overall_status == ServiceStatus.DOWN:
            logger.error("health_check_failed", status=self.status.overall_status.value,
                         errors=self.status.errors)
        elif self.status.overall_status == ServiceStatus.DEGRADED:
            logger.warning("health_check_degraded", errors=self.status.errors)

        return self.status

    def _check_ib(self) -> None:
        """Check IB connection health."""
        if self._ib_provider is None:
            self.status.ib_status = ServiceStatus.UNKNOWN
            return

        try:
            if self._ib_provider.connected:
                self.status.ib_status = ServiceStatus.UP
                self.status.ib_last_heartbeat = datetime.now()
            else:
                self.status.ib_status = ServiceStatus.DOWN
                self.status.errors.append("IB disconnected")
        except Exception as e:
            self.status.ib_status = ServiceStatus.DOWN
            self.status.errors.append(f"IB check failed: {e}")

    def _check_duckdb(self) -> None:
        """Check DuckDB is responsive."""
        if self._duckdb_conn is None:
            self.status.duckdb_status = ServiceStatus.UNKNOWN
            return

        try:
            result = self._duckdb_conn.execute("SELECT 1").fetchone()
            if result and result[0] == 1:
                self.status.duckdb_status = ServiceStatus.UP
            else:
                self.status.duckdb_status = ServiceStatus.DEGRADED
        except Exception as e:
            self.status.duckdb_status = ServiceStatus.DOWN
            self.status.errors.append(f"DuckDB check failed: {e}")

    def _check_sqlite(self) -> None:
        """Check SQLite is responsive."""
        if self._sqlite_engine is None:
            self.status.sqlite_status = ServiceStatus.UNKNOWN
            return

        try:
            with self._sqlite_engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                if result and result[0] == 1:
                    self.status.sqlite_status = ServiceStatus.UP
                else:
                    self.status.sqlite_status = ServiceStatus.DEGRADED
        except Exception as e:
            self.status.sqlite_status = ServiceStatus.DOWN
            self.status.errors.append(f"SQLite check failed: {e}")

    async def check_ollama(self, ollama_host: str = "http://localhost:11434") -> None:
        """Check if Ollama is running (async)."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_host}/api/tags")
                if resp.status_code == 200:
                    self.status.ollama_status = ServiceStatus.UP
                else:
                    self.status.ollama_status = ServiceStatus.DEGRADED
        except Exception:
            self.status.ollama_status = ServiceStatus.DOWN
