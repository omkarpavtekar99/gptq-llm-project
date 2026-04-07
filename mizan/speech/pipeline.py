"""Unified Phase 2 speech pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import mlflow

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.speech.asr import FasterWhisperAsr
from mizan.speech.diarization import PyannoteDiarization
from mizan.speech.models import (
    DiarizationSegment,
    SpeechPipelineConfig,
    SpeechPipelineMetrics,
    SpeechPipelineResult,
    SpeechSegment,
    TranscriptChunk,
    TtsSynthesisResult,
)
from mizan.speech.tts import KokoroTts
from mizan.speech.vad import SileroVadWrapper

LOGGER = get_logger(__name__)


class SpeechPipeline:
    """Run VAD, ASR, diarization, and optional TTS as one pipeline."""

    def __init__(
        self,
        settings: Settings,
        config: SpeechPipelineConfig | None = None,
        vad: SileroVadWrapper | None = None,
        asr: FasterWhisperAsr | None = None,
        diarization: PyannoteDiarization | None = None,
        tts: KokoroTts | None = None,
    ) -> None:
        """Initialize the speech pipeline."""

        self._settings = settings
        self._config = config or SpeechPipelineConfig()
        self._vad = vad or SileroVadWrapper(settings)
        self._asr = asr or FasterWhisperAsr(settings)
        self._diarization = diarization or PyannoteDiarization(settings)
        self._tts = tts or KokoroTts(settings)

    def run(
        self,
        audio_path: Path,
        reference_transcript: str | None = None,
        reference_rttm_path: Path | None = None,
        tts_text: str | None = None,
    ) -> SpeechPipelineResult:
        """Execute the enabled stages and return combined output."""

        started = time.perf_counter()
        with mlflow.start_run(
            run_name=f"speech_pipeline:{audio_path.stem}",
            nested=mlflow.active_run() is not None,
        ):
            vad_segments: list[SpeechSegment] = []
            transcript_chunks: list[TranscriptChunk] = []
            diarization_segments: list[DiarizationSegment] = []
            tts_result: TtsSynthesisResult | None = None

            if self._config.enable_vad:
                vad_segments = self._vad.detect_segments(audio_path)
            if self._config.enable_asr:
                transcript_chunks = self._asr.transcribe(audio_path, vad_segments or None)
            transcript = " ".join(chunk.text for chunk in transcript_chunks).strip()
            if self._config.enable_diarization:
                diarization_segments = self._diarization.diarize(audio_path)
            if self._config.enable_tts and tts_text:
                tts_result = self._tts.synthesize_stream(tts_text)

            wer = (
                self._asr.calculate_wer(transcript, reference_transcript)
                if reference_transcript and transcript
                else None
            )
            der = None
            if reference_rttm_path and diarization_segments:
                rttm_path = self._settings.paths.speech_output_dir / f"{audio_path.stem}.predicted.rttm"
                self._diarization.write_rttm(diarization_segments, rttm_path)
                der = self._diarization.calculate_der(rttm_path, reference_rttm_path)

            metrics = SpeechPipelineMetrics(
                total_latency_ms=round((time.perf_counter() - started) * 1000, 4),
                wer=wer,
                der=der,
                vad_segment_count=len(vad_segments),
                transcript_chunk_count=len(transcript_chunks),
                diarization_segment_count=len(diarization_segments),
            )
            self._log_metrics(metrics)
            LOGGER.info(
                "speech_pipeline_complete",
                extra={"audio_path": str(audio_path), "metrics": metrics.model_dump()},
            )
            return SpeechPipelineResult(
                audio_path=audio_path,
                transcript=transcript,
                vad_segments=vad_segments,
                transcript_chunks=transcript_chunks,
                diarization_segments=diarization_segments,
                tts_result=tts_result,
                metrics=metrics,
            )

    @staticmethod
    def _log_metrics(metrics: SpeechPipelineMetrics) -> None:
        """Log pipeline metrics to MLflow."""

        if mlflow.active_run() is None:
            return
        payload = {"speech_total_latency_ms": metrics.total_latency_ms}
        if metrics.wer is not None:
            payload["speech_wer"] = metrics.wer
        if metrics.der is not None:
            payload["speech_der"] = metrics.der
        mlflow.log_metrics(payload)
