"""Report generation for the Phase 1 benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mizan.serving.models import BenchmarkRecord, Phase1Report, QualityComparisonRecord, SweepCombination


def write_phase1_csv(path: Path, benchmark_records: list[BenchmarkRecord]) -> None:
    """Write the sweep metrics to a CSV file."""

    frame = pd.DataFrame([record.model_dump() for record in benchmark_records])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def render_phase1_summary(report: Phase1Report) -> str:
    """Render the human-readable Phase 1 Markdown summary."""

    best = report.best_record()
    selected = report.selected_config
    lines = [
        "# Phase 1 Benchmark Summary",
        "",
        "## Winning Configuration",
        f"- max_num_batched_tokens: `{selected.max_num_batched_tokens}`",
        f"- concurrent_requests: `{selected.concurrent_requests}`",
        f"- gpu_memory_utilization: `{selected.gpu_memory_utilization:.4f}`",
        f"- kv_cache_dtype: `{selected.kv_cache_dtype}`",
        f"- dtype: `{selected.dtype}`",
        f"- quantization: `{selected.quantization}`",
        "",
        "## Best Measured Sweep Result",
        f"- avg_ttft_ms: `{best.avg_ttft_ms:.4f}`",
        f"- avg_itl_ms: `{best.avg_itl_ms:.4f}`",
        f"- throughput_tokens_per_sec: `{best.throughput_tokens_per_sec:.4f}`",
        f"- peak_vram_mb: `{best.peak_vram_mb:.4f}`",
        "",
        "## Quantization Comparison",
    ]
    for quality in report.quality_records:
        lines.extend(
            [
                f"### {quality.variant_name}",
                f"- rouge_l: `{quality.rouge_l:.4f}`",
                f"- throughput_tokens_per_sec: `{quality.throughput_tokens_per_sec:.4f}`",
                f"- peak_vram_mb: `{quality.peak_vram_mb:.4f}`",
                f"- average_latency_ms: `{quality.average_latency_ms:.4f}`",
                f"- notes: {quality.notes}",
                "",
            ]
        )
    lines.extend(["## Recommendation", report.recommendation, ""])
    return "\n".join(lines)


def write_phase1_summary(path: Path, report: Phase1Report) -> None:
    """Write the Markdown summary to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase1_summary(report), encoding="utf-8")


def build_recommendation(
    best_record: BenchmarkRecord, quality_records: list[QualityComparisonRecord], selected: SweepCombination
) -> str:
    """Create the one-paragraph lock-in recommendation for future phases."""

    gpu_variant = next(record for record in quality_records if record.variant_name == "gptq_4bit_vllm")
    cpu_variant = next(record for record in quality_records if record.variant_name == "cpu_reference_transformers")
    if cpu_variant.throughput_tokens_per_sec == 0.0 and cpu_variant.average_latency_ms == 0.0:
        return (
            "Lock in the GPTQ 4-bit vLLM deployment for subsequent phases because it successfully fits "
            f"the RTX 4060 Laptop and delivered {best_record.throughput_tokens_per_sec:.4f} tokens/sec "
            f"with {best_record.avg_ttft_ms:.4f} ms TTFT at max_num_batched_tokens={selected.max_num_batched_tokens}, "
            f"concurrent_requests={selected.concurrent_requests}, and gpu_memory_utilization="
            f"{selected.gpu_memory_utilization:.4f}. The CPU reference baseline was skipped on this machine "
            f"due to host RAM limits, so the production GPTQ path should remain the locked configuration for "
            f"subsequent phases. Notes from the skipped CPU baseline: {cpu_variant.notes}"
        )
    return (
        "Lock in the GPTQ 4-bit vLLM deployment for subsequent phases because it matches the "
        f"best measured serving profile at {best_record.throughput_tokens_per_sec:.4f} tokens/sec "
        f"with {best_record.avg_ttft_ms:.4f} ms TTFT while staying within {best_record.peak_vram_mb:.4f} MB "
        f"of VRAM. The selected configuration uses max_num_batched_tokens={selected.max_num_batched_tokens}, "
        f"concurrent_requests={selected.concurrent_requests}, and gpu_memory_utilization="
        f"{selected.gpu_memory_utilization:.4f}. The CPU reference remains useful for slower quality checks "
        f"because it reached ROUGE-L {cpu_variant.rouge_l:.4f}, while the production GPTQ path achieved "
        f"ROUGE-L {gpu_variant.rouge_l:.4f} and is the best quality-to-latency fit for the RTX 4060 Laptop."
    )
