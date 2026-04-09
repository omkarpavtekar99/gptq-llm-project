"""Tests for shadow evaluation helpers."""

from __future__ import annotations

from mizan.eval.shadow import ShadowEvaluationMiddleware


def test_extract_prompt_prefers_prompt_field() -> None:
    """Shadow middleware should extract prompts from JSON requests."""

    payload = b'{"prompt":"hello world"}'

    assert ShadowEvaluationMiddleware._extract_prompt(payload) == "hello world"
