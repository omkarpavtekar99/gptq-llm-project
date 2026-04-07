"""Tests for Phase 2 ASR helpers."""

from __future__ import annotations

from config.settings import Settings
from mizan.speech.asr import FasterWhisperAsr


def test_calculate_wer_returns_zero_for_identical_strings() -> None:
    """WER should be zero when hypothesis and reference match."""

    asr = FasterWhisperAsr(Settings())

    assert asr.calculate_wer("hello world", "hello world") == 0.0
