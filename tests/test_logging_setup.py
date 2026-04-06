"""Tests for structured logging."""

from __future__ import annotations

import json
import logging

from mizan.logging_setup import JSONFormatter


def test_json_formatter_includes_standard_fields() -> None:
    """Formatter should emit the required JSON keys."""

    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="mizan.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="hello",
        args=(),
        exc_info=None,
        func="test_json_formatter_includes_standard_fields",
    )
    record.request_id = "abc123"

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["module"] == "test_logging_setup"
    assert payload["function"] == "test_json_formatter_includes_standard_fields"
    assert payload["message"] == "hello"
    assert payload["request_id"] == "abc123"


def test_get_logger_uses_shared_configuration() -> None:
    """Logger factory should return a working logger instance."""

    from mizan.logging_setup import get_logger

    logger = get_logger("mizan.test")

    assert logger.name == "mizan.test"
