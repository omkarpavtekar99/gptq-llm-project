"""Tests for Phase 2 TTS helpers."""

from __future__ import annotations

from mizan.speech.utils import percentile


def test_percentile_returns_expected_quantile() -> None:
    """Percentile helper should be stable for simple inputs."""

    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
