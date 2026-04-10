"""Root-cause comparison helpers for Phase 4."""

from __future__ import annotations

from statistics import mean

from config.settings import Settings
from mizan.analysis.models import RcaReport
from mizan.eval.engine import EvalEngine
from mizan.eval.golden_sets import GoldenSetStore
from mizan.eval.models import ModelExecutionConfig
from mizan.eval.prompt_registry import PromptRegistry


def compare_prompt_versions(settings: Settings, name: str, left_version: str, right_version: str) -> RcaReport:
    """Compare two prompt template versions by content quality delta."""

    registry = PromptRegistry(settings)
    left = registry.get(name, left_version)
    right = registry.get(name, right_version)
    left_score = _prompt_quality_score(left.content)
    right_score = _prompt_quality_score(right.content)
    winner = f"{name}:{left_version}" if left_score >= right_score else f"{name}:{right_version}"
    score_delta = round(abs(left_score - right_score), 4)
    return RcaReport(
        winner=winner,
        score_delta=score_delta,
        latency_delta=0.0,
        recommendation=f"Prefer {winner} because its prompt text scored higher on the heuristic quality comparator.",
        metadata={"left_score": left_score, "right_score": right_score},
    )


def compare_model_variants(
    settings: Settings, model_a: ModelExecutionConfig, model_b: ModelExecutionConfig, golden_name: str, golden_version: str
) -> RcaReport:
    """Compare two model configs on the same golden set."""

    store = GoldenSetStore(settings)
    engine = EvalEngine(settings)
    bundle = store.load_by_version(golden_name, golden_version)
    report_a = engine.run_eval(bundle, model_a)
    report_b = engine.run_eval(bundle, model_b)
    winner = model_a.model_name if report_a.average_judge_score >= report_b.average_judge_score else model_b.model_name
    return RcaReport(
        winner=winner,
        score_delta=round(abs(report_a.average_judge_score - report_b.average_judge_score), 4),
        latency_delta=round(
            abs(_average_latency(report_a) - _average_latency(report_b)),
            4,
        ),
        recommendation=f"Prefer {winner} because it produced the stronger average judge score on the golden set.",
        metadata={
            "model_a_judge": report_a.average_judge_score,
            "model_b_judge": report_b.average_judge_score,
        },
    )


def compare_retrieval_configs(settings: Settings, config_a: dict[str, object], config_b: dict[str, object]) -> RcaReport:
    """Compare two retrieval configs with a lightweight heuristic score."""

    score_a = _retrieval_quality_score(config_a)
    score_b = _retrieval_quality_score(config_b)
    winner = "config_a" if score_a >= score_b else "config_b"
    return RcaReport(
        winner=winner,
        score_delta=round(abs(score_a - score_b), 4),
        latency_delta=round(abs(_retrieval_latency(config_a) - _retrieval_latency(config_b)), 4),
        recommendation=f"Prefer {winner} because it offers the stronger heuristic retrieval score.",
        metadata={"config_a_score": score_a, "config_b_score": score_b},
    )


def _prompt_quality_score(content: str) -> float:
    """Heuristic prompt quality comparator for RCA simulations."""

    length_bonus = min(len(content.split()) / 40.0, 1.0)
    structure_bonus = 0.2 if "json" in content.lower() else 0.0
    ambiguity_penalty = 0.2 if "maybe" in content.lower() or "whatever" in content.lower() else 0.0
    return round(max(length_bonus + structure_bonus - ambiguity_penalty, 0.0), 4)


def _average_latency(report: object) -> float:
    """Average response latency across eval results."""

    return mean(
        [
            result.response_latency_ms or 0.0
            for result in report.results  # type: ignore[attr-defined]
        ]
    )


def _retrieval_quality_score(config: dict[str, object]) -> float:
    """Simple comparator for retrieval config RCA."""

    score = float(config.get("top_k", 3)) / 10.0
    if config.get("rerank"):
        score += 0.1
    return round(score, 4)


def _retrieval_latency(config: dict[str, object]) -> float:
    """Simple comparator for retrieval latency."""

    return round(float(config.get("top_k", 3)) * (1.5 if config.get("rerank") else 1.0), 4)
