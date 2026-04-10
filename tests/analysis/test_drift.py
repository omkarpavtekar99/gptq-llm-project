"""Tests for Phase 4 drift detection."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from mizan.analysis.drift import DriftDetector
from mizan.analysis.models import DriftBaseline
from mizan.eval.models import EvalReport, EvalResult, JudgeResult, ModelExecutionConfig


class _FakeDriftDetector(DriftDetector):
    def _embed_outputs(self, outputs: list[str]) -> list[list[float]]:
        return [[float(index), float(index + 1)] for index, _ in enumerate(outputs, start=1)]


def _report() -> EvalReport:
    return EvalReport(
        golden_set_name="demo",
        golden_set_version="1.0.0",
        execution_config=ModelExecutionConfig(model_name="demo"),
        results=[
            EvalResult(
                entry_id="e1",
                prompt="p1",
                expected_output="x",
                actual_output="y",
                tags={"domain": "serving"},
                rouge_l=0.5,
                bertscore_f1=0.6,
                judge=JudgeResult(score=4, reason="ok", latency_ms=10.0),
            ),
            EvalResult(
                entry_id="e2",
                prompt="p2",
                expected_output="x",
                actual_output="z",
                tags={"domain": "speech"},
                rouge_l=0.4,
                bertscore_f1=0.5,
                judge=JudgeResult(score=3, reason="ok", latency_ms=10.0),
            ),
        ],
        average_rouge_l=0.45,
        average_bertscore_f1=0.55,
        average_judge_score=3.5,
        error_counts={},
    )


def test_recompute_and_load_baseline(tmp_path: Path) -> None:
    """Drift baselines should round-trip through pickle storage."""

    settings = Settings()
    settings.paths.drift_baseline_path = tmp_path / "drift.pkl"
    detector = _FakeDriftDetector(settings)

    baseline = detector.recompute_baseline(_report())
    loaded = detector.load_baseline()

    assert baseline.centroid == loaded.centroid


def test_detect_returns_per_tag_scores(tmp_path: Path) -> None:
    """Drift detection should surface per-tag averages and alert metadata."""

    settings = Settings()
    settings.paths.drift_baseline_path = tmp_path / "drift.pkl"
    detector = _FakeDriftDetector(settings)
    baseline = DriftBaseline(
        centroid=[1.0, 2.0],
        covariance=[[1.0, 0.0], [0.0, 1.0]],
        judge_score_baseline=3.0,
        output_histogram=[0.05] * 20,
    )

    report = detector.detect(_report(), baseline=baseline)

    assert "domain:serving" in report.per_tag_scores
    assert "judge_delta" in report.baseline_deltas
