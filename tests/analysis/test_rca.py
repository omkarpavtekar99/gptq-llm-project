"""Tests for Phase 4 RCA helpers."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from mizan.analysis.rca import compare_prompt_versions, compare_retrieval_configs
from mizan.eval.models import PromptTemplate
from mizan.eval.prompt_registry import PromptRegistry


def test_compare_prompt_versions_prefers_more_structured_prompt(tmp_path: Path) -> None:
    """Prompt RCA should choose the stronger prompt text."""

    settings = Settings()
    settings.paths.prompt_dir = tmp_path
    registry = PromptRegistry(settings)
    registry.save(
        PromptTemplate(
            name="judge_response",
            version="1.0.0",
            content="Return strict JSON with score and reason.",
            author="omkar",
            task_type="evaluation",
        )
    )
    registry.save(
        PromptTemplate(
            name="judge_response",
            version="1.1.0",
            content="Maybe say whatever seems fine.",
            author="omkar",
            task_type="evaluation",
        )
    )

    report = compare_prompt_versions(settings, "judge_response", "1.0.0", "1.1.0")

    assert report.winner == "judge_response:1.0.0"


def test_compare_retrieval_configs_returns_a_winner() -> None:
    """Retrieval RCA should produce a deterministic recommendation."""

    report = compare_retrieval_configs(
        Settings(),
        {"top_k": 3, "rerank": False},
        {"top_k": 5, "rerank": True},
    )

    assert report.winner in {"config_a", "config_b"}
