# src/config/logging.py
from __future__ import annotations

import logging
import structlog
from src.config.settings import settings


def configure_logging() -> None:
    """Configure structlog for console/json output."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format='%(message)s',
    )
    processors = [
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.add_log_level,
    ]
    if settings.LOG_FORMAT == 'json':
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
