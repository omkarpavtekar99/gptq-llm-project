"""Server lifecycle helpers for Phase 1 vLLM benchmarks."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Sequence

import httpx

from config.settings import Settings
from mizan.logging_setup import get_logger
from mizan.serving.models import SweepCombination

LOGGER = get_logger(__name__)


class VllmServerManager:
    """Launch, stop, and health-check a local vLLM OpenAI server."""

    def __init__(self, settings: Settings) -> None:
        """Create a new server manager."""

        self._settings = settings
        self._process: subprocess.Popen[str] | None = None

    def build_command(self, combination: SweepCombination) -> list[str]:
        """Build the vLLM launch command for a sweep combination."""

        config = self._settings.vllm
        return [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            config.model_name,
            "--quantization",
            combination.quantization,
            "--gpu-memory-utilization",
            f"{combination.gpu_memory_utilization:.2f}",
            "--max-model-len",
            str(config.max_model_len),
            "--max-num-batched-tokens",
            str(combination.max_num_batched_tokens),
            "--port",
            str(config.port),
            "--dtype",
            combination.dtype,
            "--kv-cache-dtype",
            combination.kv_cache_dtype,
        ]

    def start(self, combination: SweepCombination) -> None:
        """Start the vLLM server for the given configuration."""

        self.stop()
        command = self.build_command(combination)
        LOGGER.info("starting_vllm_server", extra={"command": command, "config": combination.label})
        self._process = subprocess.Popen(  # noqa: S603
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )

    def stop(self) -> None:
        """Stop the currently running server, if one exists."""

        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        finally:
            self._process = None

    def wait_until_healthy(self) -> None:
        """Block until the server answers model-list requests."""

        deadline = time.time() + self._settings.vllm.health_timeout_sec
        models_url = f"http://127.0.0.1:{self._settings.vllm.port}/v1/models"
        last_error: str | None = None
        while time.time() < deadline:
            try:
                response = httpx.get(models_url, timeout=5.0)
                if response.status_code == 200:
                    return
                last_error = f"unexpected status {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(f"vLLM server did not become healthy: {last_error}")

    def verify_sample_completion(self, prompt: str) -> str:
        """Run a small chat completion and return the generated text."""

        payload = {
            "model": self._settings.vllm.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
        response = httpx.post(
            f"{self._settings.vllm.base_url}/chat/completions",
            json=payload,
            timeout=self._settings.thresholds.request_timeout_sec,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Unexpected health-check response payload: {body}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"Health-check response did not include message content: {body}")
        return str(content)


def build_sweep(settings: Settings) -> list[SweepCombination]:
    """Generate the full sweep grid from settings."""

    combinations: list[SweepCombination] = []
    for batched_tokens in settings.vllm.sweep_batched_tokens:
        for concurrency in settings.vllm.sweep_concurrency:
            for gpu_util in settings.vllm.sweep_gpu_memory_utilization:
                combinations.append(
                    SweepCombination(
                        max_num_batched_tokens=batched_tokens,
                        concurrent_requests=concurrency,
                        gpu_memory_utilization=gpu_util,
                        kv_cache_dtype=settings.vllm.kv_cache_dtype,
                        dtype=settings.vllm.dtype,
                        quantization=settings.vllm.quantization,
                    )
                )
    return combinations


def build_active_combination(settings: Settings) -> SweepCombination:
    """Build the currently configured live-server combination."""

    return SweepCombination(
        max_num_batched_tokens=settings.vllm.max_num_batched_tokens,
        concurrent_requests=settings.vllm.winning_concurrent_requests,
        gpu_memory_utilization=settings.vllm.gpu_memory_utilization,
        kv_cache_dtype=settings.vllm.kv_cache_dtype,
        dtype=settings.vllm.dtype,
        quantization=settings.vllm.quantization,
    )


def command_to_shell(command: Sequence[str]) -> str:
    """Render a subprocess command as a single shell string."""

    return " ".join(command)
