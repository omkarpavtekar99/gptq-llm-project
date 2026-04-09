"""Tests for the eval engine."""

from __future__ import annotations

from config.settings import Settings
from mizan.eval.engine import EvalEngine
from mizan.eval.models import GoldenSetBundle, GoldenSetEntry, JudgeResult, ModelExecutionConfig


class _FakeEvalEngine(EvalEngine):
    def _generate_response(self, prompt: str, model_config: ModelExecutionConfig) -> str:
        return "mock response"

    def _calculate_bertscore(self, prediction: str, reference: str) -> float:
        return 0.75

    def _judge_response(self, entry: GoldenSetEntry, actual_output: str) -> JudgeResult:
        return JudgeResult(score=4, reason="good", latency_ms=10.0)


def test_run_eval_returns_aggregate_report() -> None:
    """Eval reports should aggregate scored entries."""

    engine = _FakeEvalEngine(Settings())
    bundle = GoldenSetBundle(
        name="demo",
        version="1.0.0",
        entries=[
            GoldenSetEntry(
                id="g1",
                prompt="hello",
                expected_output="world",
                tags={"domain": "test"},
                version="1.0.0",
            )
        ],
    )
    report = engine.run_eval(
        bundle,
        ModelExecutionConfig(model_name="demo-model", temperature=0.0, max_tokens=32, timeout_sec=5.0),
    )

    assert report.average_judge_score == 4.0
    assert len(report.results) == 1
