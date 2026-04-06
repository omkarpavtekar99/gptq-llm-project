"""Pydantic models for Phase 1 serving benchmarks."""

from __future__ import annotations

from statistics import mean

from pydantic import BaseModel, Field


class PromptSample(BaseModel):
    """Golden prompt pair used for latency and quality evaluation."""

    prompt_id: str
    prompt: str
    expected_output: str


class SweepCombination(BaseModel):
    """One vLLM server configuration to benchmark."""

    max_num_batched_tokens: int
    concurrent_requests: int
    gpu_memory_utilization: float
    kv_cache_dtype: str = Field(default="auto")
    dtype: str = Field(default="float16")
    quantization: str = Field(default="gptq")

    @property
    def label(self) -> str:
        """Return a stable name for logs and reports."""

        return (
            "batched_tokens="
            f"{self.max_num_batched_tokens},concurrency={self.concurrent_requests},"
            f"gpu_mem={self.gpu_memory_utilization:.2f},kv_cache={self.kv_cache_dtype}"
        )


class RequestMetrics(BaseModel):
    """Per-request streaming metrics from the vLLM OpenAI endpoint."""

    prompt_id: str
    ttft_ms: float
    itl_ms: float
    throughput_tokens_per_sec: float
    total_latency_ms: float
    output_tokens: int
    output_text: str


class BenchmarkRecord(BaseModel):
    """Aggregated results for one sweep configuration."""

    max_num_batched_tokens: int
    concurrent_requests: int
    gpu_memory_utilization: float
    kv_cache_dtype: str
    dtype: str
    quantization: str
    avg_ttft_ms: float
    avg_itl_ms: float
    throughput_tokens_per_sec: float
    peak_vram_mb: float


class QualityComparisonRecord(BaseModel):
    """Quality and latency result for one model variant."""

    variant_name: str
    rouge_l: float
    throughput_tokens_per_sec: float
    peak_vram_mb: float
    average_latency_ms: float
    notes: str


class Phase1Report(BaseModel):
    """Top-level benchmark output used for CSV and Markdown rendering."""

    benchmark_records: list[BenchmarkRecord]
    quality_records: list[QualityComparisonRecord]
    selected_config: SweepCombination
    recommendation: str

    def best_record(self) -> BenchmarkRecord:
        """Return the best sweep record using latency and throughput weighting."""

        def score(record: BenchmarkRecord) -> tuple[float, float, float]:
            return (-record.throughput_tokens_per_sec, record.avg_ttft_ms, record.avg_itl_ms)

        return sorted(self.benchmark_records, key=score)[0]

    def average_rouge(self) -> float:
        """Return the mean ROUGE-L across the quality variants."""

        return mean(record.rouge_l for record in self.quality_records)
