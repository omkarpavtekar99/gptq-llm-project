"""pyannote speaker diarization integration for Phase 2."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.speech.models import DiarizationSegment

LOGGER = get_logger(__name__)


class PyannoteDiarization:
    """Speaker diarization wrapper with DER scoring."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the diarization wrapper."""

        self._settings = settings
        self._pipeline = None

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        """Return diarized speaker segments for an audio file."""

        pipeline = self._load_pipeline()
        diarization = pipeline(
            str(audio_path),
            min_speakers=self._settings.diarization.min_speakers,
            max_speakers=self._settings.diarization.max_speakers,
        )
        segments = [
            DiarizationSegment(
                speaker=str(speaker),
                start=round(float(turn.start), 4),
                end=round(float(turn.end), 4),
            )
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        LOGGER.info(
            "diarization_completed",
            extra={"audio_path": str(audio_path), "segment_count": len(segments)},
        )
        return segments

    def calculate_der(self, hypothesis_rttm: Path, reference_rttm: Path) -> float:
        """Compute diarization error rate from two RTTM files."""

        try:
            from pyannote.core import Annotation, Segment
            from pyannote.metrics.diarization import DiarizationErrorRate
        except ImportError as exc:
            raise RuntimeError("pyannote.audio and pyannote.metrics are not installed.") from exc

        def to_annotation(rttm_path: Path) -> Annotation:
            annotation = Annotation()
            for index, line in enumerate(rttm_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                parts = line.split()
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                annotation[Segment(start, start + duration), index] = speaker
            return annotation

        metric = DiarizationErrorRate()
        der = metric(to_annotation(reference_rttm), to_annotation(hypothesis_rttm))
        return round(float(der), 4)

    def write_rttm(self, segments: list[DiarizationSegment], output_path: Path) -> Path:
        """Serialize diarization output to RTTM."""

        lines = [
            f"SPEAKER audio 1 {segment.start:.4f} {segment.end - segment.start:.4f} <NA> <NA> {segment.speaker} <NA> <NA>"
            for segment in segments
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _load_pipeline(self) -> object:
        """Load the pyannote pipeline lazily."""

        if self._pipeline is not None:
            return self._pipeline
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError("pyannote.audio is not installed. Run pip install -e '.[dev]'.") from exc
        self._pipeline = Pipeline.from_pretrained(
            self._settings.diarization.model_name,
            use_auth_token=self._settings.diarization.hf_token or None,
        )
        LOGGER.info(
            "diarization_model_loaded",
            extra={"model_name": self._settings.diarization.model_name},
        )
        return self._pipeline
