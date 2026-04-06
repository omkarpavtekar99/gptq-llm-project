"""Tests for Phase 1 vLLM command generation."""

from __future__ import annotations

import httpx
import pytest

from config.settings import Settings
from mizan.serving.models import SweepCombination
from mizan.serving.vllm_runner import VllmServerManager, build_active_combination, build_sweep


def test_build_command_uses_phase1_defaults() -> None:
    """The launch command should reflect the configured sweep values."""

    settings = Settings()
    manager = VllmServerManager(settings)
    command = manager.build_command(
        SweepCombination(
            max_num_batched_tokens=2048,
            concurrent_requests=4,
            gpu_memory_utilization=0.85,
        )
    )

    assert "--model" in command
    assert "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" in command
    assert "--max-num-batched-tokens" in command
    assert "2048" in command
    assert "--gpu-memory-utilization" in command
    assert "0.85" in command


def test_build_sweep_matches_expected_grid_size() -> None:
    """The sweep should cover batched tokens, concurrency, and GPU utilization."""

    combinations = build_sweep(Settings())

    assert len(combinations) == 27


def test_build_active_combination_uses_live_server_settings() -> None:
    """The active combination should mirror the currently configured server."""

    settings = Settings()

    combination = build_active_combination(settings)

    assert combination.max_num_batched_tokens == settings.vllm.max_num_batched_tokens
    assert combination.gpu_memory_utilization == settings.vllm.gpu_memory_utilization
    assert combination.quantization == settings.vllm.quantization


def test_verify_sample_completion_raises_clear_error_on_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health check parsing should fail clearly when the response has no choices."""

    settings = Settings()
    manager = VllmServerManager(settings)

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[object]]:
            return {"choices": []}

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: DummyResponse())

    with pytest.raises(RuntimeError, match="Unexpected health-check response payload"):
        manager.verify_sample_completion("hello")
