"""Utility helpers for Phase 2 speech processing."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from statistics import mean

import numpy as np


def load_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and normalize it to mono float32."""

    try:
        import soundfile as sf

        samples, sample_rate = sf.read(str(audio_path), always_2d=False, dtype="float32")
        if isinstance(samples, np.ndarray) and samples.ndim > 1:
            samples = samples.mean(axis=1)
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    except ImportError:
        pass
    except Exception:
        pass

    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    dtype_map: dict[int, np.dtype[np.generic]] = {1: np.int8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    samples = np.frombuffer(frames, dtype=dtype_map[sample_width]).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    scale = float(2 ** (8 * sample_width - 1))
    return samples / scale, sample_rate


def write_wav_mono(audio_path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples to a 16-bit PCM WAV file."""

    clipped = np.clip(samples, -1.0, 1.0)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(audio_path), clipped.astype(np.float32), sample_rate, subtype="PCM_16")
        return
    except ImportError:
        pass
    except Exception:
        pass

    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def slice_audio(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Extract a time slice from a waveform."""

    start_index = max(int(start * sample_rate), 0)
    end_index = min(int(end * sample_rate), len(samples))
    return samples[start_index:end_index]


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio with linear interpolation."""

    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration_sec = len(samples) / float(source_rate)
    source_times = np.linspace(0.0, duration_sec, num=len(samples), endpoint=False)
    target_length = max(int(round(duration_sec * target_rate)), 1)
    target_times = np.linspace(0.0, duration_sec, num=target_length, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32)


def compute_energy_db(samples: np.ndarray) -> float:
    """Compute segment loudness in dBFS."""

    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 0.0:
        return -120.0
    return 20 * math.log10(rms)


def percentile(values: list[float], quantile: float) -> float:
    """Compute one percentile using NumPy for stable benchmarking output."""

    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def average_or_none(values: list[float | None]) -> float | None:
    """Average non-null values, returning None when none are present."""

    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def round_metric(value: float | None) -> float | None:
    """Round numeric metrics to four decimals."""

    return None if value is None else round(value, 4)
