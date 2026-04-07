"""faster-whisper ASR integration for Phase 2."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import mlflow

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.speech.models import SpeechSegment, TranscriptChunk
from mizan.speech.utils import load_wav_mono, slice_audio, write_wav_mono

LOGGER = get_logger(__name__)


class FasterWhisperAsr:
    """ASR wrapper supporting streamed chunks and WER scoring."""

    def __init__(self, settings: Settings, model_size: str | None = None) -> None:
        """Initialize the ASR wrapper."""

        self._settings = settings
        self._model_size = model_size or settings.asr.model_size
        self._model = None

    def stream_transcription(
        self, audio_path: Path, segments: list[SpeechSegment] | None = None
    ) -> Iterator[TranscriptChunk]:
        """Yield transcription chunks as they arrive."""

        model = self._load_model()
        waveform, sample_rate = load_wav_mono(audio_path)
        work_segments = segments or [
            SpeechSegment(start=0.0, end=round(len(waveform) / sample_rate, 4))
        ]
        for segment in work_segments:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            try:
                write_wav_mono(
                    temp_path,
                    slice_audio(waveform, sample_rate, segment.start, segment.end),
                    sample_rate,
                )
                started = time.perf_counter()
                segment_generator, _ = model.transcribe(
                    str(temp_path),
                    language=self._settings.asr.language,
                    beam_size=self._settings.asr.beam_size,
                    vad_filter=False,
                    word_timestamps=False,
                )
                first_chunk_at: float | None = None
                for result in segment_generator:
                    now = time.perf_counter()
                    if first_chunk_at is None:
                        first_chunk_at = now
                    chunk = TranscriptChunk(
                        text=result.text.strip(),
                        start=round(segment.start + float(result.start), 4),
                        end=round(segment.start + float(result.end), 4),
                        ttft_ms=round((first_chunk_at - started) * 1000, 4),
                        total_processing_ms=round((now - started) * 1000, 4),
                    )
                    if mlflow.active_run() is not None:
                        mlflow.log_metric("asr_segment_ttft_ms", chunk.ttft_ms)
                    yield chunk
                if mlflow.active_run() is not None:
                    mlflow.log_metric(
                        "asr_segment_total_ms", round((time.perf_counter() - started) * 1000, 4)
                    )
            finally:
                temp_path.unlink(missing_ok=True)

    def transcribe(
        self, audio_path: Path, segments: list[SpeechSegment] | None = None
    ) -> list[TranscriptChunk]:
        """Return all transcript chunks for an audio file."""

        return list(self.stream_transcription(audio_path, segments))

    def calculate_wer(self, hypothesis: str, reference: str) -> float:
        """Compute WER for one hypothesis/reference pair."""

        try:
            from jiwer import wer
        except ImportError as exc:
            raise RuntimeError("jiwer is not installed. Run pip install -e '.[dev]'.") from exc
        return round(float(wer(reference, hypothesis)), 4)

    def _load_model(self) -> object:
        """Load the faster-whisper model lazily."""

        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed. Run pip install -e '.[dev]'.") from exc
        self._model = WhisperModel(
            self._model_size,
            device=self._settings.asr.device,
            compute_type=self._settings.asr.compute_type,
        )
        LOGGER.info("asr_model_loaded", extra={"model_size": self._model_size})
        return self._model
