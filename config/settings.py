"""Application settings for Mizan."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    """Filesystem locations used across the project."""

    project_root: Path = Field(default=Path("."))
    data_dir: Path = Field(default=Path("data"))
    results_dir: Path = Field(default=Path("results"))
    reports_dir: Path = Field(default=Path("reports"))
    prompt_dir: Path = Field(default=Path("config/prompts"))
    golden_set_dir: Path = Field(default=Path("data/golden_sets"))
    baseline_dir: Path = Field(default=Path("data/baselines"))
    shadow_db_path: Path = Field(default=Path("data/shadow_log.db"))
    phase1_benchmark_csv: Path = Field(default=Path("results/phase1_benchmark.csv"))
    phase1_summary_md: Path = Field(default=Path("results/phase1_summary.md"))

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "PathsConfig":
        """Resolve all relative paths against the project root."""

        root = self.project_root.resolve()
        self.project_root = root
        self.data_dir = self._resolve_from_root(root, self.data_dir)
        self.results_dir = self._resolve_from_root(root, self.results_dir)
        self.reports_dir = self._resolve_from_root(root, self.reports_dir)
        self.prompt_dir = self._resolve_from_root(root, self.prompt_dir)
        self.golden_set_dir = self._resolve_from_root(root, self.golden_set_dir)
        self.baseline_dir = self._resolve_from_root(root, self.baseline_dir)
        self.shadow_db_path = self._resolve_from_root(root, self.shadow_db_path)
        self.phase1_benchmark_csv = self._resolve_from_root(root, self.phase1_benchmark_csv)
        self.phase1_summary_md = self._resolve_from_root(root, self.phase1_summary_md)
        return self

    @staticmethod
    def _resolve_from_root(root: Path, value: Path) -> Path:
        """Resolve a path relative to the project root when needed."""

        return value if value.is_absolute() else (root / value)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    json_output: bool = Field(default=True, validation_alias=AliasChoices("json_output", "json"))


class VllmConfig(BaseModel):
    """vLLM runtime configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    base_url: str = Field(default="http://127.0.0.1:8000/v1")
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    quantization: str = Field(default="gptq")
    dtype: str = Field(default="float16")
    gpu_memory_utilization: float = Field(default=0.99)
    max_model_len: int = Field(default=1024)
    max_num_batched_tokens: int = Field(default=512)
    kv_cache_dtype: str = Field(default="auto")
    cpu_offload_gb: float = Field(default=0.0)
    benchmark_timeout_sec: int = Field(default=120)
    health_timeout_sec: int = Field(default=180)
    sweep_batched_tokens: list[int] = Field(default_factory=lambda: [512, 1024, 2048])
    sweep_concurrency: list[int] = Field(default_factory=lambda: [1, 4, 8])
    sweep_gpu_memory_utilization: list[float] = Field(default_factory=lambda: [0.80, 0.85, 0.90])
    winning_max_num_batched_tokens: int = Field(default=512)
    winning_concurrent_requests: int = Field(default=4)
    winning_gpu_memory_utilization: float = Field(default=0.99)
    winning_quantization: str = Field(default="gptq")
    winning_dtype: str = Field(default="float16")
    cpu_reference_model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    cpu_reference_device: str = Field(default="cpu")
    cpu_reference_dtype: str = Field(default="float32")
    enable_cpu_reference: bool = Field(default=False)
    eval_max_tokens: int = Field(default=160)
    warmup_prompts: int = Field(default=2)
    prompt_sample_size: int = Field(default=20)

    @field_validator("sweep_batched_tokens", "sweep_concurrency", mode="before")
    @classmethod
    def _parse_int_list(cls, value: object) -> object:
        """Parse comma-separated integer lists from environment values."""

        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return value

    @field_validator("sweep_gpu_memory_utilization", mode="before")
    @classmethod
    def _parse_float_list(cls, value: object) -> object:
        """Parse comma-separated float lists from environment values."""

        if isinstance(value, str):
            return [float(item.strip()) for item in value.split(",") if item.strip()]
        return value


class MlflowConfig(BaseModel):
    """MLflow tracking configuration."""

    tracking_uri: str = Field(default="file:./mlruns")
    experiment_name: str = Field(default="gptq-llm-project")


class PrometheusConfig(BaseModel):
    """Prometheus exposition configuration."""

    port: int = Field(default=9090)


class ThresholdConfig(BaseModel):
    """Cross-phase quality and alert thresholds."""

    rouge_min: float = Field(default=0.35)
    judge_min: float = Field(default=3.8)
    hallucination_max_rate: float = Field(default=0.05)
    drift_alert_delta: float = Field(default=0.05)
    request_timeout_sec: float = Field(default=5.0)


class VadConfig(BaseModel):
    """Voice activity detection thresholds."""

    threshold: float = Field(default=0.50)
    min_silence_ms: int = Field(default=250)
    energy_gate_db: float = Field(default=-40.0)


class AsrConfig(BaseModel):
    """Automatic speech recognition runtime settings."""

    model_size: str = Field(default="medium")
    device: str = Field(default="cuda")
    compute_type: str = Field(default="float16")


class DiarizationConfig(BaseModel):
    """Speaker diarization settings."""

    model_name: str = Field(default="pyannote/speaker-diarization-3.1")
    hf_token: str = Field(default="")


class TtsConfig(BaseModel):
    """Text-to-speech settings."""

    model_name: str = Field(default="kokoro")
    voice: str = Field(default="af_bella")
    stream_chunk_size: int = Field(default=1024)


class RoutingConfig(BaseModel):
    """Model routing defaults."""

    primary_model_alias: str = Field(default="qwen-gptq")
    secondary_model_alias: str = Field(default="qwen-gptq-safe")
    cache_ttl_sec: int = Field(default=300)
    strict_latency_sla_ms: int = Field(default=2000)


class RagConfig(BaseModel):
    """Retrieval and embedding settings."""

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_dir: Path = Field(default=Path("data/chroma"))

class Settings(BaseSettings):
    """Top-level application settings loaded from environment variables."""

    app_env: str = Field(default="development")
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    rag: RagConfig = Field(default_factory=RagConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @model_validator(mode="after")
    def resolve_runtime_paths(self) -> "Settings":
        """Resolve cross-section paths after settings load."""

        if not self.rag.chroma_persist_dir.is_absolute():
            self.rag.chroma_persist_dir = self.paths.project_root / self.rag.chroma_persist_dir
        return self

    def ensure_directories(self) -> None:
        """Create required project directories if they do not exist."""

        directories = (
            self.paths.project_root,
            self.paths.data_dir,
            self.paths.results_dir,
            self.paths.reports_dir,
            self.paths.prompt_dir,
            self.paths.golden_set_dir,
            self.paths.baseline_dir,
            self.rag.chroma_persist_dir,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
