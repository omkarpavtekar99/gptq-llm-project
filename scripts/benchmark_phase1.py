"""Phase 1 benchmark entrypoint for Mizan."""

from __future__ import annotations

import argparse
import asyncio

import mlflow

from config.settings import get_settings
from mizan.logging_setup import get_logger
from mizan.serving.benchmark import (
    StreamingBenchmark,
    configure_mlflow,
    log_quality_record,
    log_sweep_record,
)
from mizan.serving.models import Phase1Report, SweepCombination
from mizan.serving.prompt_library import get_phase1_prompts
from mizan.serving.reporting import build_recommendation, write_phase1_csv, write_phase1_summary
from mizan.serving.vllm_runner import (
    VllmServerManager,
    build_active_combination,
    build_sweep,
    command_to_shell,
)

LOGGER = get_logger(__name__)


async def run_phase1(manage_server: bool) -> None:
    """Execute the full Phase 1 serving benchmark."""

    settings = get_settings()
    settings.ensure_directories()
    prompts = get_phase1_prompts(settings.vllm.prompt_sample_size)
    benchmark = StreamingBenchmark(settings)
    server = VllmServerManager(settings)
    sweep_records = []
    combinations = build_sweep(settings) if manage_server else [build_active_combination(settings)]

    with configure_mlflow(settings, run_name="phase1_serving_benchmark"):
        mlflow.log_param("prompt_sample_size", settings.vllm.prompt_sample_size)
        mlflow.log_param("manage_server", manage_server)
        mlflow.log_param("combination_count", len(combinations))
        for combination in combinations:
            try:
                if manage_server:
                    server.start(combination)
                    server.wait_until_healthy()
                health_text = server.verify_sample_completion("Reply with the word healthy.")
                LOGGER.info(
                    "health_check_passed",
                    extra={"config": combination.label, "sample": health_text},
                )

                with mlflow.start_run(run_name=combination.label, nested=True):
                    record = await benchmark.benchmark_combination(combination, prompts)
                    log_sweep_record(record)
                sweep_records.append(record)
            except Exception as exc:
                LOGGER.warning(
                    "sweep_combination_failed",
                    extra={"config": combination.label, "error": str(exc)},
                )
                if manage_server:
                    server.stop()
                continue

        if not sweep_records:
            if manage_server:
                raise RuntimeError("No benchmark sweep combinations completed successfully.")
            raise RuntimeError(
                "The running vLLM server was not reachable at "
                f"{settings.vllm.base_url}. Keep `make serve` running in another terminal "
                "or rerun with `--manage-server`."
            )

        quality_records = await benchmark.evaluate_quality(prompts)
        with mlflow.start_run(run_name="quantization_comparison", nested=True):
            for record in quality_records:
                log_quality_record(record)

        selected = SweepCombination(
            max_num_batched_tokens=settings.vllm.winning_max_num_batched_tokens,
            concurrent_requests=settings.vllm.winning_concurrent_requests,
            gpu_memory_utilization=settings.vllm.winning_gpu_memory_utilization,
            kv_cache_dtype=settings.vllm.kv_cache_dtype,
            dtype=settings.vllm.winning_dtype,
            quantization=settings.vllm.winning_quantization,
        )
        best_record = sorted(
            sweep_records,
            key=lambda item: (-item.throughput_tokens_per_sec, item.avg_ttft_ms, item.avg_itl_ms),
        )[0]
        recommendation = build_recommendation(best_record, quality_records, selected)
        report = Phase1Report(
            benchmark_records=sweep_records,
            quality_records=quality_records,
            selected_config=selected,
            recommendation=recommendation,
        )
        write_phase1_csv(settings.paths.phase1_benchmark_csv, sweep_records)
        write_phase1_summary(settings.paths.phase1_summary_md, report)
        mlflow.log_artifact(str(settings.paths.phase1_benchmark_csv))
        mlflow.log_artifact(str(settings.paths.phase1_summary_md))
        LOGGER.info(
            "phase1_benchmark_complete",
            extra={
                "csv_path": str(settings.paths.phase1_benchmark_csv),
                "summary_path": str(settings.paths.phase1_summary_md),
                "selected_profile": selected.label,
                "recommended_command": command_to_shell(server.build_command(selected)),
            },
        )
    server.stop()


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Phase 1 benchmark."""

    parser = argparse.ArgumentParser(description="Run the Mizan Phase 1 vLLM benchmark.")
    parser.add_argument(
        "--manage-server",
        action="store_true",
        help="Restart the local vLLM server for each sweep combination.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(run_phase1(manage_server=cli_args.manage_server))
