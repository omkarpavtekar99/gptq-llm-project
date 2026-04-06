"""Tests for Phase 1 benchmark helpers."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from mizan.serving.benchmark import log_quality_record, log_sweep_record
from mizan.serving.models import BenchmarkRecord, PromptSample, QualityComparisonRecord


def test_log_helpers_accept_phase1_records(monkeypatch: MonkeyPatch) -> None:
    """MLflow logging helpers should accept the Phase 1 record models."""

    logged_metrics: dict[str, float] = {}
    logged_params: dict[str, object] = {}

    def capture_metrics(payload: dict[str, float]) -> None:
        logged_metrics.update(payload)

    def capture_params(payload: dict[str, object]) -> None:
        logged_params.update(payload)

    monkeypatch.setattr("mlflow.log_metrics", capture_metrics)
    monkeypatch.setattr("mlflow.log_params", capture_params)

    sweep_record = BenchmarkRecord(
        max_num_batched_tokens=2048,
        concurrent_requests=4,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="auto",
        dtype="float16",
        quantization="gptq",
        avg_ttft_ms=125.0,
        avg_itl_ms=18.0,
        throughput_tokens_per_sec=44.0,
        peak_vram_mb=4812.0,
    )
    quality_record = QualityComparisonRecord(
        variant_name="gptq_4bit_vllm",
        rouge_l=0.88,
        throughput_tokens_per_sec=44.0,
        peak_vram_mb=4812.0,
        average_latency_ms=900.0,
        notes="Primary.",
    )

    log_sweep_record(sweep_record)
    log_quality_record(quality_record)

    assert logged_params["max_num_batched_tokens"] == 2048
    assert logged_metrics["avg_ttft_ms"] == 125.0
    assert logged_metrics["gptq_4bit_vllm_rouge_l"] == 0.88


@pytest.mark.asyncio
async def test_stream_prompt_raises_clear_error_when_stream_is_empty(monkeypatch: MonkeyPatch) -> None:
    """Streaming parser should fail clearly when no tokens are returned."""

    from config.settings import Settings
    from mizan.serving.benchmark import StreamingBenchmark

    settings = Settings()
    benchmark = StreamingBenchmark(settings)

    class DummyStream:
        async def __aenter__(self) -> "DummyStream":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self) -> object:
            if False:
                yield ""

    class DummyClient:
        async def __aenter__(self) -> "DummyClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def stream(self, *args: object, **kwargs: object) -> DummyStream:
            return DummyStream()

    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: DummyClient())

    with pytest.raises(RuntimeError, match="completed without any tokens"):
        await benchmark._stream_prompt(
            PromptSample(prompt_id="p01", prompt="hi", expected_output="hello")
        )


@pytest.mark.asyncio
async def test_evaluate_quality_skips_cpu_reference_on_load_failure(monkeypatch: MonkeyPatch) -> None:
    """CPU reference failures should not abort the full quality pass."""

    from config.settings import Settings
    from mizan.serving.benchmark import StreamingBenchmark
    from mizan.serving.models import PromptSample, QualityComparisonRecord

    settings = Settings()
    benchmark = StreamingBenchmark(settings)

    async def fake_gpu_eval(prompts: list[PromptSample]) -> QualityComparisonRecord:
        return QualityComparisonRecord(
            variant_name="gptq_4bit_vllm",
            rouge_l=0.9,
            throughput_tokens_per_sec=40.0,
            peak_vram_mb=4800.0,
            average_latency_ms=900.0,
            notes="Primary.",
        )

    def fake_cpu_eval(prompts: list[PromptSample]) -> QualityComparisonRecord:
        raise RuntimeError("Cannot allocate memory")

    monkeypatch.setattr(benchmark, "_evaluate_gptq", fake_gpu_eval)
    monkeypatch.setattr(benchmark, "_evaluate_cpu_reference", fake_cpu_eval)

    results = await benchmark.evaluate_quality(
        [PromptSample(prompt_id="p01", prompt="hi", expected_output="hello")]
    )

    assert len(results) == 2
    assert results[1].variant_name == "cpu_reference_transformers"
    assert results[1].throughput_tokens_per_sec == 0.0
    assert "could not load" in results[1].notes


@pytest.mark.asyncio
async def test_evaluate_quality_skips_cpu_reference_when_disabled(monkeypatch: MonkeyPatch) -> None:
    """CPU reference should be skipped immediately when disabled in settings."""

    from config.settings import Settings
    from mizan.serving.benchmark import StreamingBenchmark
    from mizan.serving.models import PromptSample, QualityComparisonRecord

    settings = Settings()
    settings.vllm.enable_cpu_reference = False
    benchmark = StreamingBenchmark(settings)

    async def fake_gpu_eval(prompts: list[PromptSample]) -> QualityComparisonRecord:
        return QualityComparisonRecord(
            variant_name="gptq_4bit_vllm",
            rouge_l=0.9,
            throughput_tokens_per_sec=40.0,
            peak_vram_mb=4800.0,
            average_latency_ms=900.0,
            notes="Primary.",
        )

    monkeypatch.setattr(benchmark, "_evaluate_gptq", fake_gpu_eval)

    results = await benchmark.evaluate_quality(
        [PromptSample(prompt_id="p01", prompt="hi", expected_output="hello")]
    )

    assert len(results) == 2
    assert results[1].variant_name == "cpu_reference_transformers"
    assert results[1].throughput_tokens_per_sec == 0.0
    assert "disabled" in results[1].notes
