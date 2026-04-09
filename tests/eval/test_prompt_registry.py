"""Tests for the prompt registry."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from mizan.eval.models import PromptTemplate
from mizan.eval.prompt_registry import PromptRegistry


def test_save_and_diff_prompt_versions(tmp_path: Path) -> None:
    """The registry should save templates and diff versions cleanly."""

    settings = Settings()
    settings.paths.prompt_dir = tmp_path
    registry = PromptRegistry(settings)
    registry.save(
        PromptTemplate(
            name="demo",
            version="1.0.0",
            content="Line one",
            author="omkar",
            task_type="test",
        )
    )
    registry.save(
        PromptTemplate(
            name="demo",
            version="1.1.0",
            content="Line one\nLine two",
            author="omkar",
            task_type="test",
        )
    )

    diff = registry.diff("demo", "1.0.0", "1.1.0")

    assert "Line two" in diff
