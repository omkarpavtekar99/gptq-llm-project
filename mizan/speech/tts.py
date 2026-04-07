"""Kokoro TTS integration for Phase 2."""

from __future__ import annotations

import time

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.speech.models import LatencySummary, TtsSynthesisResult
from mizan.speech.utils import percentile

LOGGER = get_logger(__name__)


class KokoroTts:
    """Streaming TTS wrapper around Kokoro."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the TTS wrapper."""

        self._settings = settings
        self._pipeline = None

    def synthesize_stream(self, text: str, voice: str | None = None) -> TtsSynthesisResult:
        """Run streaming TTS and return latency percentiles."""

        pipeline = self._load_pipeline()
        selected_voice = voice or self._settings.tts.voice
        latencies_ms: list[float] = []
        sample_count = 0
        started = time.perf_counter()
        chunk_count = 0
        for _, _, audio in pipeline(
            text,
            voice=selected_voice,
            speed=self._settings.tts.speed,
        ):
            chunk_count += 1
            latencies_ms.append(round((time.perf_counter() - started) * 1000, 4))
            sample_count += len(audio)
        summary = LatencySummary(
            p50_ms=round(percentile(latencies_ms, 50), 4),
            p95_ms=round(percentile(latencies_ms, 95), 4),
            p99_ms=round(percentile(latencies_ms, 99), 4),
        )
        result = TtsSynthesisResult(
            voice=selected_voice,
            sample_rate=self._settings.tts.sample_rate,
            chunk_count=chunk_count,
            audio_duration_sec=round(sample_count / self._settings.tts.sample_rate, 4),
            latency_summary=summary,
        )
        LOGGER.info("tts_completed", extra={"voice": selected_voice, "chunk_count": chunk_count})
        return result

    def _load_pipeline(self) -> object:
        """Load the Kokoro pipeline lazily."""

        if self._pipeline is not None:
            return self._pipeline
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("kokoro is not installed. Run pip install -e '.[dev]'.") from exc
        self._pipeline = KPipeline(lang_code=self._settings.tts.lang_code)
        return self._pipeline
