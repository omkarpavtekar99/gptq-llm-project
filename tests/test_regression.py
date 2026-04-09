"""Pytest regression harness for Phase 3."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from mizan.eval.engine import EvalEngine
from mizan.eval.golden_sets import GoldenSetStore
from mizan.eval.models import (
    EvalReport,
    EvalResult,
    GoldenSetBundle,
    JudgeResult,
    ModelExecutionConfig,
)
from mizan.eval.regression import (
    assert_report_within_thresholds,
    baseline_from_report,
    load_regression_baseline,
    save_regression_baseline,
)


class _RegressionEvalEngine(EvalEngine):
    def run_eval(self, golden_set: GoldenSetBundle, model_config: ModelExecutionConfig) -> EvalReport:
        results = [
            EvalResult(
                entry_id=entry.id,
                prompt=entry.prompt,
                expected_output=entry.expected_output,
                actual_output=entry.expected_output,
                rouge_l=0.85,
                bertscore_f1=0.82,
                judge=JudgeResult(score=4, reason="fixture", latency_ms=8.0),
            )
            for entry in golden_set.entries
        ]
        return EvalReport(
            golden_set_name=golden_set.name,
            golden_set_version=golden_set.version,
            execution_config=model_config,
            results=results,
            average_rouge_l=0.85,
            average_bertscore_f1=0.82,
            average_judge_score=4.0,
            error_counts={},
        )


def test_regression_suite(pytestconfig: pytest.Config, tmp_path: Path) -> None:
    """Fail when rolling quality drops below configured thresholds."""

    settings = Settings()
    settings.paths.golden_set_dir = tmp_path / "golden_sets"
    settings.paths.regression_baseline_path = tmp_path / "regression_baseline.json"
    store = GoldenSetStore(settings)
    source_bundle = GoldenSetStore(Settings()).load_by_version("core_eval", "1.0.0")
    store.publish(source_bundle.name, source_bundle.version, source_bundle.entries)
    engine = _RegressionEvalEngine(settings)
    report = engine.run_eval(
        store.load_by_version("core_eval", "1.0.0"),
        ModelExecutionConfig(model_name=settings.vllm.model_name),
    )
    current = baseline_from_report(report)
    if pytestconfig.getoption("--update-baseline") or not settings.paths.regression_baseline_path.exists():
        save_regression_baseline(settings, current)
        pytest.skip("Regression baseline updated.")
    baseline = load_regression_baseline(settings)
    assert_report_within_thresholds(settings, report, baseline)
