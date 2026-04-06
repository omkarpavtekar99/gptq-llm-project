"""Tests for the Phase 1 prompt library."""

from __future__ import annotations

from mizan.serving.prompt_library import get_phase1_prompts


def test_phase1_prompt_library_has_expected_sample_size() -> None:
    """The prompt library should expose the requested 20 prompts."""

    prompts = get_phase1_prompts()

    assert len(prompts) == 20
    assert prompts[0].prompt_id == "p01"
