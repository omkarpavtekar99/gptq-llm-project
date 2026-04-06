"""Tests for settings loading."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings


def test_settings_load_nested_environment(monkeypatch: object) -> None:
    """Nested environment variables should override defaults."""

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("VLLM__PORT", "9001")
    monkeypatch.setenv("VLLM__GPU_MEMORY_UTILIZATION", "0.9")
    monkeypatch.setenv("PATHS__PROJECT_ROOT", "/tmp/mizan")

    settings = Settings()

    assert settings.app_env == "test"
    assert settings.vllm.port == 9001
    assert settings.vllm.gpu_memory_utilization == 0.9
    assert settings.paths.project_root == Path("/tmp/mizan")


def test_ensure_directories_creates_expected_paths(monkeypatch: object, tmp_path: Path) -> None:
    """Directory bootstrap should create the configured folders."""

    monkeypatch.setenv("PATHS__PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("PATHS__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PATHS__RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("PATHS__REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("PATHS__PROMPT_DIR", str(tmp_path / "config" / "prompts"))
    monkeypatch.setenv("PATHS__GOLDEN_SET_DIR", str(tmp_path / "data" / "golden_sets"))
    monkeypatch.setenv("PATHS__BASELINE_DIR", str(tmp_path / "data" / "baselines"))
    monkeypatch.setenv("RAG__CHROMA_PERSIST_DIR", str(tmp_path / "data" / "chroma"))

    settings = Settings()
    settings.ensure_directories()

    assert (tmp_path / "data").exists()
    assert (tmp_path / "results").exists()
    assert (tmp_path / "config" / "prompts").exists()
    assert (tmp_path / "data" / "chroma").exists()


def test_phase1_sweep_lists_parse_from_env(monkeypatch: object) -> None:
    """Comma-separated Phase 1 sweep values should parse into numeric lists."""

    monkeypatch.setenv("VLLM__SWEEP_BATCHED_TOKENS", "512,1024,2048")
    monkeypatch.setenv("VLLM__SWEEP_CONCURRENCY", "1,4,8")
    monkeypatch.setenv("VLLM__SWEEP_GPU_MEMORY_UTILIZATION", "0.80,0.85,0.90")

    settings = Settings()

    assert settings.vllm.sweep_batched_tokens == [512, 1024, 2048]
    assert settings.vllm.sweep_concurrency == [1, 4, 8]
    assert settings.vllm.sweep_gpu_memory_utilization == [0.8, 0.85, 0.9]
