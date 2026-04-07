"""Shared data models for the Phase 2 speech stack."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SpeechSegment(BaseModel):
    """A speech interval in seconds."""

    start: float
    end: float
    energy_db: float | None = None


class TranscriptChunk(BaseModel):
    """One streamed ASR chunk."""

    text: str
    start: float
    end: float
    ttft_ms: float
    total_processing_ms: float


class DiarizationSegment(BaseModel):
    """A diarized speaker interval."""

    speaker: str
    start: float
    end: float


class LatencySummary(BaseModel):
    """Latency percentiles in milliseconds."""

    p50_ms: float
    p95_ms: float
    p99_ms: float


class TtsSynthesisResult(BaseModel):
    """Streaming TTS metadata."""

    voice: str
    sample_rate: int
    chunk_count: int
    audio_duration_sec: float
    latency_summary: LatencySummary


class SpeechPipelineConfig(BaseModel):
    """Runtime toggles for the unified speech pipeline."""

    enable_vad: bool = Field(default=True)
    enable_asr: bool = Field(default=True)
    enable_diarization: bool = Field(default=True)
    enable_tts: bool = Field(default=False)


class SpeechPipelineMetrics(BaseModel):
    """Aggregated metrics emitted by the speech pipeline."""

    total_latency_ms: float
    wer: float | None = None
    der: float | None = None
    vad_segment_count: int = 0
    transcript_chunk_count: int = 0
    diarization_segment_count: int = 0


class SpeechPipelineResult(BaseModel):
    """End-to-end speech pipeline output."""

    audio_path: Path
    transcript: str
    vad_segments: list[SpeechSegment]
    transcript_chunks: list[TranscriptChunk]
    diarization_segments: list[DiarizationSegment]
    tts_result: TtsSynthesisResult | None = None
    metrics: SpeechPipelineMetrics


class SpeechBenchmarkSample(BaseModel):
    """Manifest entry for the Phase 2 benchmark."""

    sample_id: str
    audio_path: Path
    reference_transcript: str
    reference_rttm_path: Path | None = None
    notes: str = ""


class SpeechBenchmarkRecord(BaseModel):
    """Per-sample benchmark record."""

    model_size: str
    sample_id: str
    wer: float
    der: float | None
    total_latency_ms: float
    vram_delta_mb: float


class SpeechBenchmarkAggregate(BaseModel):
    """Aggregate metrics for one ASR model size."""

    model_size: str
    avg_wer: float
    avg_der: float | None
    avg_total_latency_ms: float
    avg_vram_delta_mb: float


class SpeechBenchmarkReport(BaseModel):
    """Phase 2 benchmark report."""

    records: list[SpeechBenchmarkRecord]
    aggregates: list[SpeechBenchmarkAggregate]
    recommendation: str
