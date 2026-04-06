"""Tests for Phase 1 report rendering."""

from __future__ import annotations

from mizan.serving.models import BenchmarkRecord, Phase1Report, QualityComparisonRecord, SweepCombination
from mizan.serving.reporting import build_recommendation, render_phase1_summary


def test_render_phase1_summary_contains_winning_configuration() -> None:
    """The summary should surface the selected winning configuration."""

    selected = SweepCombination(
        max_num_batched_tokens=2048,
        concurrent_requests=4,
        gpu_memory_utilization=0.85,
    )
    benchmark_record = BenchmarkRecord(
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
    quality_records = [
        QualityComparisonRecord(
            variant_name="gptq_4bit_vllm",
            rouge_l=0.88,
            throughput_tokens_per_sec=44.0,
            peak_vram_mb=4812.0,
            average_latency_ms=900.0,
            notes="Primary.",
        ),
        QualityComparisonRecord(
            variant_name="cpu_reference_transformers",
            rouge_l=0.90,
            throughput_tokens_per_sec=1.2,
            peak_vram_mb=0.0,
            average_latency_ms=6000.0,
            notes="Reference.",
        ),
    ]
    report = Phase1Report(
        benchmark_records=[benchmark_record],
        quality_records=quality_records,
        selected_config=selected,
        recommendation=build_recommendation(benchmark_record, quality_records, selected),
    )

    summary = render_phase1_summary(report)

    assert "Winning Configuration" in summary
    assert "2048" in summary
    assert "gptq_4bit_vllm" in summary
