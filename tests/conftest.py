"""Shared pytest options for the Mizan test suite."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register project-specific pytest flags."""

    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help="Update the regression baseline instead of asserting against it.",
    )
