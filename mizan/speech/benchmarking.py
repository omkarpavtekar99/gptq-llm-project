"""Phase 2 speech benchmark orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import mlflow
import pandas as pd
from pynvml import NVMLError, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.speech.asr import FasterWhisperAsr
from mizan.speech.models import (
    SpeechBenchmarkAggregate,
    SpeechBenchmarkRecord,
    SpeechBenchmarkReport,
    SpeechBenchmarkSample,
    SpeechPipelineConfig,
)
from mizan.speech.pipeline import SpeechPipeline
from mizan.speech.utils import average_or_none, round_metric

LOGGER = get_logger(__name__)


def load_manifest(manifest_path: Path) -> list[SpeechBenchmarkSample]:
    """Load the benchmark sample manifest."""

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Benchmark manifest not found: {manifest_path}. Copy data/phase2_benchmark_manifest.example.json first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [SpeechBenchmarkSample.model_validate(item) for item in payload]
    for sample in samples:
        if not sample.audio_path.is_absolute():
            sample.audio_path = (manifest_path.parent / sample.audio_path).resolve()
        if sample.reference_rttm_path is not None and not sample.reference_rttm_path.is_absolute():
            sample.reference_rttm_path = (manifest_path.parent / sample.reference_rttm_path).resolve()
    return samples


def run_benchmark(settings: Settings, manifest_path: Path | None = None) -> SpeechBenchmarkReport:
    """Run Phase 2 benchmark across configured Whisper model sizes."""

    manifest = load_manifest(manifest_path or settings.paths.phase2_benchmark_manifest)
    if not manifest:
        raise ValueError("Phase 2 benchmark manifest is empty.")
    records: list[SpeechBenchmarkRecord] = []
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    enable_diarization = any(sample.reference_rttm_path is not None for sample in manifest) and bool(
        settings.diarization.hf_token
    )
    if not enable_diarization:
        LOGGER.warning(
            "phase2_diarization_disabled",
            extra={"reason": "Missing RTTM references or DIARIZATION__HF_TOKEN."},
        )
    with mlflow.start_run(run_name="phase2_speech_benchmark"):
        for model_size in settings.asr.benchmark_model_sizes:
            pipeline = SpeechPipeline(
                settings,
                config=SpeechPipelineConfig(enable_tts=False, enable_diarization=enable_diarization),
                asr=FasterWhisperAsr(settings, model_size=model_size),
            )
            with mlflow.start_run(run_name=f"phase2:{model_size}", nested=True):
                for sample in manifest:
                    before = read_vram_mb()
                    result = pipeline.run(
                        sample.audio_path,
                        reference_transcript=sample.reference_transcript,
                        reference_rttm_path=sample.reference_rttm_path,
                    )
                    after = read_vram_mb()
                    record = SpeechBenchmarkRecord(
                        model_size=model_size,
                        sample_id=sample.sample_id,
                        wer=round(result.metrics.wer or 0.0, 4),
                        der=round_metric(result.metrics.der),
                        total_latency_ms=round(result.metrics.total_latency_ms, 4),
                        vram_delta_mb=round(max(after - before, 0.0), 4),
                    )
                    records.append(record)
        aggregates = build_aggregates(records)
        report = SpeechBenchmarkReport(
            records=records,
            aggregates=aggregates,
            recommendation=build_recommendation(aggregates),
        )
        write_outputs(settings, report)
        mlflow.log_artifact(str(settings.paths.phase2_benchmark_csv))
        mlflow.log_artifact(str(settings.paths.phase2_summary_md))
        return report


def build_aggregates(records: list[SpeechBenchmarkRecord]) -> list[SpeechBenchmarkAggregate]:
    """Aggregate per-sample benchmark records by model size."""

    model_sizes = sorted({record.model_size for record in records})
    aggregates: list[SpeechBenchmarkAggregate] = []
    for model_size in model_sizes:
        selected = [record for record in records if record.model_size == model_size]
        aggregates.append(
            SpeechBenchmarkAggregate(
                model_size=model_size,
                avg_wer=round(mean(item.wer for item in selected), 4),
                avg_der=round_metric(average_or_none([item.der for item in selected])),
                avg_total_latency_ms=round(mean(item.total_latency_ms for item in selected), 4),
                avg_vram_delta_mb=round(mean(item.vram_delta_mb for item in selected), 4),
            )
        )
    return aggregates


def build_recommendation(aggregates: list[SpeechBenchmarkAggregate]) -> str:
    """Recommend the best ASR model tradeoff from benchmark aggregates."""

    best = sorted(aggregates, key=lambda item: (item.avg_wer, item.avg_total_latency_ms))[0]
    return (
        f"Use Whisper {best.model_size} as the default Phase 2 ASR profile because it achieved "
        f"average WER {best.avg_wer:.4f} with average end-to-end latency {best.avg_total_latency_ms:.4f} ms "
        f"and average VRAM delta {best.avg_vram_delta_mb:.4f} MB on the benchmark manifest."
    )


def write_outputs(settings: Settings, report: SpeechBenchmarkReport) -> None:
    """Write CSV and Markdown outputs for Phase 2."""

    settings.paths.phase2_benchmark_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.model_dump() for record in report.records]).to_csv(
        settings.paths.phase2_benchmark_csv, index=False
    )
    lines = ["# Phase 2 Speech Benchmark Summary", ""]
    for aggregate in report.aggregates:
        lines.extend(
            [
                f"## {aggregate.model_size}",
                f"- avg_wer: `{aggregate.avg_wer:.4f}`",
                f"- avg_der: `{'n/a' if aggregate.avg_der is None else f'{aggregate.avg_der:.4f}'}`",
                f"- avg_total_latency_ms: `{aggregate.avg_total_latency_ms:.4f}`",
                f"- avg_vram_delta_mb: `{aggregate.avg_vram_delta_mb:.4f}`",
                "",
            ]
        )
    lines.extend(["## Recommendation", report.recommendation, ""])
    settings.paths.phase2_summary_md.write_text("\n".join(lines), encoding="utf-8")


def read_vram_mb() -> float:
    """Read current VRAM usage in MB."""

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        return nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
    except NVMLError as exc:
        LOGGER.warning("phase2_nvml_read_failed", extra={"error": str(exc)})
        return 0.0
