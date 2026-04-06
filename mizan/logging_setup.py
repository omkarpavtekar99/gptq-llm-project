"""Structured JSON logging helpers for Mizan."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

STANDARD_LOG_RECORD_FIELDS: set[str] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JSONFormatter(logging.Formatter):
    """Format log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record to JSON."""

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        extras = self._extract_extras(record)
        payload.update(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)

    def _extract_extras(self, record: logging.LogRecord) -> dict[str, Any]:
        """Return non-standard attributes added through logger extras."""

        return {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_LOG_RECORD_FIELDS and not key.startswith("_")
        }


def configure_logging(level: str | None = None) -> None:
    """Configure root logging with the shared JSON formatter."""

    effective_level = (level or os.getenv("LOGGING__LEVEL", "INFO")).upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    if any(getattr(handler, "_mizan_handler", False) for handler in root_logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    handler._mizan_handler = True  # type: ignore[attr-defined]
    root_logger.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with Mizan's JSON formatter."""

    configure_logging()
    return logging.getLogger(name)
