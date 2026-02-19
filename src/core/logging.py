"""Structured logging setup using structlog."""

import logging
import os
import sys

import structlog

from src.config import PROJECT_ROOT, settings


def _safe_log_output():
    """Return a safe file handle for structlog output.

    On Windows, Streamlit can replace sys.stderr with an object that
    raises [Errno 22] Invalid argument on write.  We detect this and
    fall back to a log file or devnull.
    """
    # Quick test: can we actually write to stderr?
    try:
        sys.stderr.write("")
        sys.stderr.flush()
        return sys.stderr
    except (OSError, AttributeError, ValueError):
        pass

    # Fallback: write to a log file
    try:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        return open(log_dir / "trading.log", "a", encoding="utf-8")  # noqa: SIM115
    except Exception:
        return open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def setup_logging() -> None:
    """Configure structlog for JSON or console output."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log.level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=_safe_log_output()),
        cache_logger_on_first_use=False,
    )


# Auto-configure on first import so all loggers get safe output
setup_logging()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)
