"""Benchmark orchestration for Phase 1."""

from __future__ import annotations

import asyncio
import json
import time
from statistics import mean
from typing import Any

import httpx
import mlflow
from pynvml import NVMLError, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.serving.models import (
    BenchmarkRecord,
    PromptSample,
    QualityComparisonRecord,
    RequestMetrics,
    SweepCombination,
)

LOGGER = get_logger(__name__)


class StreamingBenchmark:
    """Run the Phase 1 parameter sweep and quantization comparison."""

    def __init__(self, settings: Settings) -> None:
        """Create a benchmark runner."""

        self._settings = settings
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    async def benchmark_combination(
        self, combination: SweepCombination, prompts: list[PromptSample]
    ) -> BenchmarkRecord:
        """Measure a single sweep configuration against multiple prompts."""

        await self._warmup(prompts[: self._settings.vllm.warmup_prompts])
        batches = self._build_prompt_batches(prompts, combination.concurrent_requests)
        request_results: list[RequestMetrics] = []
        for batch in batches:
            results = await asyncio.gather(*[self._stream_prompt(prompt) for prompt in batch])
            request_results.extend(results)

        return BenchmarkRecord(
            max_num_batched_tokens=combination.max_num_batched_tokens,
            concurrent_requests=combination.concurrent_requests,
            gpu_memory_utilization=round(combination.gpu_memory_utilization, 4),
            kv_cache_dtype=combination.kv_cache_dtype,
            dtype=combination.dtype,
            quantization=combination.quantization,
            avg_ttft_ms=round(mean(item.ttft_ms for item in request_results), 4),
            avg_itl_ms=round(mean(item.itl_ms for item in request_results), 4),
            throughput_tokens_per_sec=round(
                mean(item.throughput_tokens_per_sec for item in request_results), 4
            ),
            peak_vram_mb=round(self._read_peak_vram_mb(), 4),
        )

    async def evaluate_quality(
        self, prompts: list[PromptSample]
    ) -> list[QualityComparisonRecord]:
        """Run quality comparison for GPTQ and CPU-reference variants."""

        gpu_metrics = await self._evaluate_gptq(prompts)
        if not self._settings.vllm.enable_cpu_reference:
            cpu_metrics = QualityComparisonRecord(
                variant_name="cpu_reference_transformers",
                rouge_l=0.0,
                throughput_tokens_per_sec=0.0,
                peak_vram_mb=0.0,
                average_latency_ms=0.0,
                notes=(
                    "Skipped on this machine because VLLM__ENABLE_CPU_REFERENCE is disabled. "
                    "This avoids OOM risk on low-RAM WSL setups."
                ),
            )
            return [gpu_metrics, cpu_metrics]
        try:
            cpu_metrics = self._evaluate_cpu_reference(prompts)
        except Exception as exc:
            LOGGER.warning("cpu_reference_failed", extra={"error": str(exc)})
            cpu_metrics = QualityComparisonRecord(
                variant_name="cpu_reference_transformers",
                rouge_l=0.0,
                throughput_tokens_per_sec=0.0,
                peak_vram_mb=0.0,
                average_latency_ms=0.0,
                notes=f"Skipped on this machine because the CPU reference model could not load: {exc}",
            )
        return [gpu_metrics, cpu_metrics]

    async def _evaluate_gptq(self, prompts: list[PromptSample]) -> QualityComparisonRecord:
        """Evaluate the live GPTQ server."""

        request_results = [await self._stream_prompt(prompt) for prompt in prompts]
        rouge_scores = [
            self._scorer.score(prompt.expected_output, result.output_text)["rougeL"].fmeasure
            for prompt, result in zip(prompts, request_results, strict=True)
        ]
        return QualityComparisonRecord(
            variant_name="gptq_4bit_vllm",
            rouge_l=round(mean(rouge_scores), 4),
            throughput_tokens_per_sec=round(
                mean(result.throughput_tokens_per_sec for result in request_results), 4
            ),
            peak_vram_mb=round(self._read_peak_vram_mb(), 4),
            average_latency_ms=round(mean(result.total_latency_ms for result in request_results), 4),
            notes="Primary benchmarked vLLM deployment.",
        )

    def _evaluate_cpu_reference(self, prompts: list[PromptSample]) -> QualityComparisonRecord:
        """Evaluate a CPU reference model through Transformers."""

        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(self._settings.vllm.cpu_reference_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self._settings.vllm.cpu_reference_model_name,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        rouge_scores: list[float] = []
        latencies_ms: list[float] = []
        token_rates: list[float] = []
        for prompt in prompts:
            prompt_started = time.perf_counter()
            generated = generator(
                prompt.prompt,
                max_new_tokens=self._settings.vllm.eval_max_tokens,
                return_full_text=False,
                do_sample=False,
            )
            latency_ms = (time.perf_counter() - prompt_started) * 1000
            output_text = str(generated[0]["generated_text"])
            output_tokens = max(len(output_text.split()), 1)
            rouge_scores.append(
                self._scorer.score(prompt.expected_output, output_text)["rougeL"].fmeasure
            )
            latencies_ms.append(latency_ms)
            token_rates.append(output_tokens / max(latency_ms / 1000, 0.001))

        total_duration_ms = (time.perf_counter() - started) * 1000
        return QualityComparisonRecord(
            variant_name="cpu_reference_transformers",
            rouge_l=round(mean(rouge_scores), 4),
            throughput_tokens_per_sec=round(mean(token_rates), 4),
            peak_vram_mb=0.0,
            average_latency_ms=round(mean(latencies_ms), 4),
            notes=f"CPU reference over {len(prompts)} prompts, wall time {total_duration_ms:.4f} ms.",
        )

    async def _warmup(self, prompts: list[PromptSample]) -> None:
        """Warm up the server before measurements."""

        for prompt in prompts:
            await self._stream_prompt(prompt)

    def _build_prompt_batches(
        self, prompts: list[PromptSample], concurrency: int
    ) -> list[list[PromptSample]]:
        """Split prompts into equally sized concurrency batches."""

        return [prompts[index : index + concurrency] for index in range(0, len(prompts), concurrency)]

    async def _stream_prompt(self, prompt: PromptSample) -> RequestMetrics:
        """Send a single streaming chat completion and measure timing."""

        payload = {
            "model": self._settings.vllm.model_name,
            "messages": [{"role": "user", "content": prompt.prompt}],
            "temperature": 0.0,
            "max_tokens": self._settings.vllm.eval_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        timestamps: list[float] = []
        output_chunks: list[str] = []
        output_tokens = 0
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._settings.vllm.benchmark_timeout_sec) as client:
            async with client.stream(
                "POST",
                f"{self._settings.vllm.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if not raw_line.startswith("data: ") or raw_line == "data: [DONE]":
                        continue
                    body = json.loads(raw_line.removeprefix("data: "))
                    choices = body.get("choices", [])
                    if not isinstance(choices, list) or not choices:
                        if "error" in body:
                            raise RuntimeError(f"Streaming response returned error payload: {body['error']}")
                        continue
                    timestamps.append(time.perf_counter())
                    if usage := body.get("usage"):
                        output_tokens = int(usage.get("completion_tokens", output_tokens))
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        output_chunks.append(str(delta))

        output_text = "".join(output_chunks).strip()
        if not output_text and not timestamps:
            raise RuntimeError("Streaming response completed without any tokens.")
        if output_tokens == 0:
            output_tokens = max(len(output_text.split()), 1)
        ttft_ms = max((timestamps[0] - started) * 1000, 0.0) if timestamps else 0.0
        inter_token_gaps = [
            (timestamps[index] - timestamps[index - 1]) * 1000 for index in range(1, len(timestamps))
        ]
        total_latency_ms = max((time.perf_counter() - started) * 1000, 0.0)
        return RequestMetrics(
            prompt_id=prompt.prompt_id,
            ttft_ms=round(ttft_ms, 4),
            itl_ms=round(mean(inter_token_gaps), 4) if inter_token_gaps else 0.0,
            throughput_tokens_per_sec=round(output_tokens / max(total_latency_ms / 1000, 0.001), 4),
            total_latency_ms=round(total_latency_ms, 4),
            output_tokens=output_tokens,
            output_text=output_text,
        )

    def _read_peak_vram_mb(self) -> float:
        """Read the current GPU memory usage in MB through NVML."""

        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            memory = nvmlDeviceGetMemoryInfo(handle)
            return memory.used / (1024 * 1024)
        except NVMLError as exc:
            LOGGER.warning("nvml_read_failed", extra={"error": str(exc)})
            return 0.0


def configure_mlflow(settings: Settings, run_name: str) -> mlflow.ActiveRun:
    """Start an MLflow run for Phase 1."""

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_sweep_record(record: BenchmarkRecord) -> None:
    """Log one sweep record to MLflow."""

    mlflow.log_metrics(
        {
            "avg_ttft_ms": record.avg_ttft_ms,
            "avg_itl_ms": record.avg_itl_ms,
            "throughput_tokens_per_sec": record.throughput_tokens_per_sec,
            "peak_vram_mb": record.peak_vram_mb,
        }
    )
    mlflow.log_params(
        {
            "max_num_batched_tokens": record.max_num_batched_tokens,
            "concurrent_requests": record.concurrent_requests,
            "gpu_memory_utilization": record.gpu_memory_utilization,
            "kv_cache_dtype": record.kv_cache_dtype,
            "dtype": record.dtype,
            "quantization": record.quantization,
        }
    )


def log_quality_record(record: QualityComparisonRecord) -> None:
    """Log one quality comparison record to MLflow."""

    metrics: dict[str, Any] = {
        f"{record.variant_name}_rouge_l": record.rouge_l,
        f"{record.variant_name}_throughput_tokens_per_sec": record.throughput_tokens_per_sec,
        f"{record.variant_name}_peak_vram_mb": record.peak_vram_mb,
        f"{record.variant_name}_average_latency_ms": record.average_latency_ms,
    }
    mlflow.log_metrics(metrics)
