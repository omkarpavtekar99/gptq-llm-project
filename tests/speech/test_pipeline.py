"""Tests for the unified speech pipeline."""

from __future__ import annotations

from pathlib import Path

import mlflow

from config.settings import Settings
from mizan.speech.models import (
    DiarizationSegment,
    LatencySummary,
    SpeechPipelineConfig,
    SpeechSegment,
    TranscriptChunk,
    TtsSynthesisResult,
)
from mizan.speech.pipeline import SpeechPipeline


class _FakeVad:
    def detect_segments(self, audio_path: Path) -> list[SpeechSegment]:
        return [SpeechSegment(start=0.0, end=1.0, energy_db=-15.0)]


class _FakeAsr:
    def transcribe(
        self, audio_path: Path, segments: list[SpeechSegment] | None = None
    ) -> list[TranscriptChunk]:
        return [TranscriptChunk(text="hello", start=0.0, end=1.0, ttft_ms=1.0, total_processing_ms=2.0)]

    def calculate_wer(self, hypothesis: str, reference: str) -> float:
        return 0.25


class _FakeDiarization:
    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        return [DiarizationSegment(speaker="spk1", start=0.0, end=1.0)]

    def write_rttm(self, segments: list[DiarizationSegment], output_path: Path) -> Path:
        output_path.write_text("SPEAKER audio 1 0.0 1.0 <NA> <NA> spk1 <NA> <NA>", encoding="utf-8")
        return output_path

    def calculate_der(self, hypothesis_rttm: Path, reference_rttm: Path) -> float:
        return 0.1


class _FakeTts:
    def synthesize_stream(self, text: str, voice: str | None = None) -> TtsSynthesisResult:
        return TtsSynthesisResult(
            voice=voice or "af_bella",
            sample_rate=24000,
            chunk_count=1,
            audio_duration_sec=0.5,
            latency_summary=LatencySummary(p50_ms=1.0, p95_ms=1.0, p99_ms=1.0),
        )


def test_pipeline_runs_with_injected_stages(tmp_path: Path) -> None:
    """The pipeline should aggregate outputs from each stage."""

    settings = Settings()
    settings.paths.speech_output_dir = tmp_path
    reference_rttm = tmp_path / "reference.rttm"
    reference_rttm.write_text("SPEAKER audio 1 0.0 1.0 <NA> <NA> spk1 <NA> <NA>", encoding="utf-8")
    with mlflow.start_run():
        pipeline = SpeechPipeline(
            settings,
            config=SpeechPipelineConfig(enable_tts=True),
            vad=_FakeVad(),
            asr=_FakeAsr(),
            diarization=_FakeDiarization(),
            tts=_FakeTts(),
        )
        result = pipeline.run(
            tmp_path / "sample.wav",
            reference_transcript="hello there",
            reference_rttm_path=reference_rttm,
            tts_text="hello there",
        )

    assert result.transcript == "hello"
    assert result.metrics.wer == 0.25
    assert result.metrics.der == 0.1
    assert result.tts_result is not None
