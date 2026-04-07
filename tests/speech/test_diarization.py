"""Tests for Phase 2 diarization helpers."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from mizan.speech.diarization import PyannoteDiarization
from mizan.speech.models import DiarizationSegment


def test_write_rttm_serializes_segments(tmp_path: Path) -> None:
    """RTTM export should write one line per diarization segment."""

    wrapper = PyannoteDiarization(Settings())
    output = wrapper.write_rttm(
        [DiarizationSegment(speaker="spk1", start=0.0, end=1.5)],
        tmp_path / "sample.rttm",
    )

    content = output.read_text(encoding="utf-8")

    assert "SPEAKER audio 1 0.0000 1.5000" in content
