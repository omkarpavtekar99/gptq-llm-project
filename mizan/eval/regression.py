"""Regression baseline helpers for Phase 3."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import Settings
from mizan.eval.models import ErrorClass, EvalReport, RegressionBaseline


def baseline_from_report(report: EvalReport) -> RegressionBaseline:
    """Convert an eval report into a persisted regression baseline."""

    hallucinations = sum(1 for result in report.results if result.error_class == ErrorClass.hallucination)
    rate = hallucinations / max(len(report.results), 1)
    return RegressionBaseline(
        average_judge_score=report.average_judge_score,
        average_rouge_l=report.average_rouge_l,
        hallucination_rate=round(rate, 4),
    )


def load_regression_baseline(settings: Settings) -> RegressionBaseline:
    """Load the persisted regression baseline."""

    path = settings.paths.regression_baseline_path
    return RegressionBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def save_regression_baseline(settings: Settings, baseline: RegressionBaseline) -> Path:
    """Persist the regression baseline to disk."""

    path = settings.paths.regression_baseline_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def assert_report_within_thresholds(
    settings: Settings, report: EvalReport, baseline: RegressionBaseline
) -> None:
    """Raise an AssertionError when the current report regresses too far."""

    current = baseline_from_report(report)
    if current.average_judge_score < settings.thresholds.judge_min:
        raise AssertionError(
            f"Average judge score dropped to {current.average_judge_score:.4f}, below "
            f"threshold {settings.thresholds.judge_min:.4f}."
        )
    minimum_rouge = baseline.average_rouge_l * (1 - settings.thresholds.rouge_drop_tolerance)
    if current.average_rouge_l < minimum_rouge:
        raise AssertionError(
            f"Average ROUGE-L dropped to {current.average_rouge_l:.4f}, below "
            f"baseline floor {minimum_rouge:.4f}."
        )
    if current.hallucination_rate > settings.thresholds.hallucination_max_rate:
        raise AssertionError(
            f"Hallucination rate rose to {current.hallucination_rate:.4f}, above "
            f"limit {settings.thresholds.hallucination_max_rate:.4f}."
        )
