"""Tests for regression baseline helpers."""

from __future__ import annotations

from config.settings import Settings
from mizan.eval.models import (
    EvalReport,
    EvalResult,
    JudgeResult,
    ModelExecutionConfig,
    RegressionBaseline,
)
from mizan.eval.regression import assert_report_within_thresholds, baseline_from_report


def test_baseline_from_report_computes_hallucination_rate() -> None:
    """Regression baselines should derive aggregate metrics from reports."""

    report = EvalReport(
        golden_set_name="demo",
        golden_set_version="1.0.0",
        execution_config=ModelExecutionConfig(model_name="demo"),
        results=[
            EvalResult(
                entry_id="e1",
                prompt="p",
                expected_output="e",
                actual_output="a",
                rouge_l=0.8,
                bertscore_f1=0.7,
                judge=JudgeResult(score=4, reason="ok", latency_ms=5.0),
            )
        ],
        average_rouge_l=0.8,
        average_bertscore_f1=0.7,
        average_judge_score=4.0,
        error_counts={},
    )

    baseline = baseline_from_report(report)

    assert baseline.average_judge_score == 4.0


def test_assert_report_within_thresholds_accepts_good_report() -> None:
    """Good reports should pass the regression checks."""

    settings = Settings()
    report = EvalReport(
        golden_set_name="demo",
        golden_set_version="1.0.0",
        execution_config=ModelExecutionConfig(model_name="demo"),
        results=[
            EvalResult(
                entry_id="e1",
                prompt="p",
                expected_output="e",
                actual_output="a",
                rouge_l=0.8,
                bertscore_f1=0.7,
                judge=JudgeResult(score=4, reason="ok", latency_ms=5.0),
            )
        ],
        average_rouge_l=0.8,
        average_bertscore_f1=0.7,
        average_judge_score=4.0,
        error_counts={},
    )
    baseline = RegressionBaseline(average_judge_score=4.0, average_rouge_l=0.8, hallucination_rate=0.0)

    assert_report_within_thresholds(settings, report, baseline)
