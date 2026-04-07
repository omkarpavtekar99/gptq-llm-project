"""Tests for Phase 2 speech benchmarking."""

from __future__ import annotations

from mizan.speech.benchmarking import build_aggregates, build_recommendation
from mizan.speech.models import SpeechBenchmarkRecord


def test_build_aggregates_groups_records_by_model_size() -> None:
    """Aggregates should average records per ASR model size."""

    aggregates = build_aggregates(
        [
            SpeechBenchmarkRecord(
                model_size="medium",
                sample_id="s1",
                wer=0.2,
                der=0.1,
                total_latency_ms=100.0,
                vram_delta_mb=200.0,
            ),
            SpeechBenchmarkRecord(
                model_size="large-v3",
                sample_id="s1",
                wer=0.1,
                der=0.05,
                total_latency_ms=150.0,
                vram_delta_mb=350.0,
            ),
        ]
    )

    assert len(aggregates) == 2
    assert build_recommendation(aggregates).startswith("Use Whisper")
