"""Tests for Phase 2 VAD helpers."""

from __future__ import annotations

from config.settings import Settings
from mizan.speech.models import SpeechSegment
from mizan.speech.vad import SileroVadWrapper


def test_merge_short_gaps_merges_adjacent_segments() -> None:
    """Short silent gaps should be merged into one segment."""

    settings = Settings()
    wrapper = SileroVadWrapper(settings)
    merged = wrapper.merge_short_gaps(
        [
            SpeechSegment(start=0.0, end=1.0, energy_db=-20.0),
            SpeechSegment(start=1.1, end=2.0, energy_db=-18.0),
            SpeechSegment(start=3.0, end=4.0, energy_db=-15.0),
        ]
    )

    assert len(merged) == 2
    assert merged[0].start == 0.0
    assert merged[0].end == 2.0
