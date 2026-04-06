"""Tests for Phase 1 serving data models."""

from __future__ import annotations

from mizan.serving.models import BenchmarkRecord, Phase1Report, QualityComparisonRecord, SweepCombination


def test_phase1_report_best_record_prefers_higher_throughput() -> None:
    """The report should choose the record with the highest throughput."""

    fast_record = BenchmarkRecord(
        max_num_batched_tokens=2048,
        concurrent_requests=4,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="auto",
        dtype="float16",
        quantization="gptq",
        avg_ttft_ms=120.0,
        avg_itl_ms=15.0,
        throughput_tokens_per_sec=50.0,
        peak_vram_mb=4800.0,
    )
    slow_record = BenchmarkRecord(
        max_num_batched_tokens=1024,
        concurrent_requests=4,
        gpu_memory_utilization=0.80,
        kv_cache_dtype="auto",
        dtype="float16",
        quantization="gptq",
        avg_ttft_ms=110.0,
        avg_itl_ms=14.0,
        throughput_tokens_per_sec=30.0,
        peak_vram_mb=4300.0,
    )
    report = Phase1Report(
        benchmark_records=[slow_record, fast_record],
        quality_records=[
            QualityComparisonRecord(
                variant_name="gptq_4bit_vllm",
                rouge_l=0.9,
                throughput_tokens_per_sec=50.0,
                peak_vram_mb=4800.0,
                average_latency_ms=850.0,
                notes="Primary.",
            )
        ],
        selected_config=SweepCombination(
            max_num_batched_tokens=2048,
            concurrent_requests=4,
            gpu_memory_utilization=0.85,
        ),
        recommendation="Use the faster record.",
    )

    assert report.best_record() == fast_record
