"""Silero VAD integration for Phase 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.speech.models import SpeechSegment
from mizan.speech.utils import compute_energy_db, load_wav_mono, resample_audio, slice_audio

LOGGER = get_logger(__name__)


class SileroVadWrapper:
    """Detect speech segments in WAV audio using Silero VAD."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the wrapper with shared settings."""

        self._settings = settings
        self._model = None

    def detect_segments(self, audio_path: Path) -> list[SpeechSegment]:
        """Return filtered speech segments for a WAV file."""

        samples, sample_rate = load_wav_mono(audio_path)
        if sample_rate not in {8000, 16000}:
            target_rate = self._settings.vad.sampling_rate
            samples = resample_audio(samples, sample_rate, target_rate)
            sample_rate = target_rate
        model = self._load_model()
        timestamps = self._get_speech_timestamps(samples, model, sample_rate, audio_path)
        segments = [
            SpeechSegment(start=float(item["start"]), end=float(item["end"]))
            for item in timestamps
        ]
        gated_segments = self.energy_gate(samples, sample_rate, segments)
        return self.merge_short_gaps(gated_segments)

    def energy_gate(
        self, samples: np.ndarray, sample_rate: int, segments: list[SpeechSegment]
    ) -> list[SpeechSegment]:
        """Drop segments whose energy falls below the configured gate."""

        kept: list[SpeechSegment] = []
        for segment in segments:
            window = slice_audio(samples, sample_rate, segment.start, segment.end)
            energy_db = compute_energy_db(window)
            if energy_db >= self._settings.vad.energy_gate_db:
                kept.append(
                    SpeechSegment(start=segment.start, end=segment.end, energy_db=round(energy_db, 4))
                )
        return kept

    def merge_short_gaps(self, segments: list[SpeechSegment]) -> list[SpeechSegment]:
        """Merge adjacent speech regions separated by short silence."""

        if not segments:
            return []
        merged: list[SpeechSegment] = [segments[0]]
        max_gap_sec = self._settings.vad.min_silence_ms / 1000.0
        for current in segments[1:]:
            previous = merged[-1]
            if current.start - previous.end <= max_gap_sec:
                merged[-1] = SpeechSegment(
                    start=previous.start,
                    end=current.end,
                    energy_db=max(
                        [value for value in (previous.energy_db, current.energy_db) if value is not None],
                        default=None,
                    ),
                )
            else:
                merged.append(current)
        return merged

    def _load_model(self) -> object:
        """Load the Silero VAD model lazily."""

        if self._model is not None:
            return self._model
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError("silero-vad is not installed. Run pip install -e '.[dev]'.") from exc
        self._model = load_silero_vad()
        return self._model

    def _get_speech_timestamps(
        self, samples: np.ndarray, model: object, sample_rate: int, audio_path: Path
    ) -> list[dict[str, float]]:
        """Execute Silero's speech timestamp detection."""

        try:
            from silero_vad import get_speech_timestamps
        except ImportError as exc:
            raise RuntimeError("silero-vad is not installed. Run pip install -e '.[dev]'.") from exc
        timestamps = get_speech_timestamps(
            samples,
            model,
            threshold=self._settings.vad.threshold,
            min_speech_duration_ms=self._settings.vad.min_speech_duration_ms,
            min_silence_duration_ms=self._settings.vad.min_silence_ms,
            speech_pad_ms=self._settings.vad.speech_pad_ms,
            sampling_rate=sample_rate,
            return_seconds=True,
        )
        LOGGER.info(
            "vad_segments_detected",
            extra={"audio_path": str(audio_path), "segment_count": len(timestamps)},
        )
        return timestamps
