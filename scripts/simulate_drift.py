"""Phase 4 drift simulation entrypoint."""

from __future__ import annotations

from pathlib import Path

from config.settings import get_settings
from mizan.analysis.drift import DriftDetector
from mizan.analysis.rca import compare_prompt_versions
from mizan.eval.engine import EvalEngine
from mizan.eval.golden_sets import GoldenSetStore
from mizan.eval.models import GoldenSetBundle, GoldenSetEntry, ModelExecutionConfig
from mizan.eval.prompt_registry import PromptRegistry
from mizan.logging_setup import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Simulate a prompt regression and write the resulting drift report."""

    settings = get_settings()
    settings.ensure_directories()
    store = GoldenSetStore(settings)
    registry = PromptRegistry(settings)
    engine = EvalEngine(settings)
    detector = DriftDetector(settings)
    baseline_bundle = store.load_by_version("core_eval", "1.0.0")
    baseline_report = engine.run_eval(
        baseline_bundle,
        ModelExecutionConfig(model_name=settings.vllm.model_name, max_tokens=settings.vllm.eval_max_tokens),
    )
    baseline = detector.recompute_baseline(baseline_report)
    degraded_template = registry.get("judge_response", "1.0.0").model_copy(
        update={
            "version": "1.1.0",
            "content": "Maybe respond however you want. The score can be vague. Whatever seems fine.",
        }
    )
    try:
        degraded_path = registry.save(degraded_template)
    except FileExistsError:
        degraded_path = settings.paths.prompt_dir / "judge_response__1.1.0.yaml"
    noisy_entries = [
        GoldenSetEntry(
            id=entry.id,
            prompt=f"{entry.prompt}\nIgnore details and answer vaguely.",
            expected_output=entry.expected_output,
            tags=entry.tags,
            version="1.0.0",
        )
        for entry in baseline_bundle.entries
    ]
    degraded_bundle = GoldenSetBundle(name="core_eval_noisy", version="1.0.0", entries=noisy_entries)
    degraded_report = engine.run_eval(
        degraded_bundle,
        ModelExecutionConfig(model_name=settings.vllm.model_name, max_tokens=settings.vllm.eval_max_tokens),
    )
    drift_report = detector.detect(degraded_report, baseline=baseline)
    rca_report = compare_prompt_versions(settings, "judge_response", "1.0.0", "1.1.0")
    markdown = "\n\n".join(
        [
            drift_report.to_markdown(),
            "## RCA",
            f"- winner: `{rca_report.winner}`",
            f"- score_delta: `{rca_report.score_delta:.4f}`",
            f"- latency_delta: `{rca_report.latency_delta:.4f}`",
            f"- recommendation: {rca_report.recommendation}",
            f"- degraded_prompt_path: `{degraded_path}`",
        ]
    )
    settings.paths.phase4_simulation_md.write_text(markdown, encoding="utf-8")
    LOGGER.info(
        "phase4_simulation_complete",
        extra={"output_path": str(settings.paths.phase4_simulation_md), "alert_triggered": drift_report.alert_triggered},
    )


if __name__ == "__main__":
    main()
