"""Shared data models for the Phase 3 evaluation framework."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ErrorClass(str, Enum):
    """Supported error taxonomy for evaluation failures."""

    timeout = "timeout"
    low_quality = "low_quality"
    hallucination = "hallucination"
    refusal = "refusal"
    format_error = "format_error"


class GoldenSetEntry(BaseModel):
    """One immutable golden-set example."""

    id: str
    prompt: str
    expected_output: str
    tags: dict[str, str]
    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GoldenSetBundle(BaseModel):
    """A published golden set version stored as one JSON file."""

    name: str
    version: str
    entries: list[GoldenSetEntry]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_path: Path | None = None


class PromptTemplate(BaseModel):
    """One versioned prompt template."""

    name: str
    version: str
    content: str
    author: str
    task_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)


class ModelExecutionConfig(BaseModel):
    """Model request parameters for eval-time generation."""

    model_name: str
    temperature: float = 0.0
    max_tokens: int = 160
    timeout_sec: float = 60.0


class JudgeResult(BaseModel):
    """Structured output from the LLM-as-judge path."""

    score: int
    reason: str
    latency_ms: float


class EvalResult(BaseModel):
    """Scored output for one golden-set entry."""

    entry_id: str
    prompt: str
    expected_output: str
    actual_output: str
    tags: dict[str, str] = Field(default_factory=dict)
    rouge_l: float
    bertscore_f1: float
    judge: JudgeResult
    error_class: ErrorClass | None = None
    response_latency_ms: float | None = None


class EvalReport(BaseModel):
    """Aggregate report for one eval run."""

    golden_set_name: str
    golden_set_version: str
    execution_config: ModelExecutionConfig
    results: list[EvalResult]
    average_rouge_l: float
    average_bertscore_f1: float
    average_judge_score: float
    error_counts: dict[str, int]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RagEvalResult(BaseModel):
    """RAG eval result for one question."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class RagEvalReport(BaseModel):
    """Aggregate RAG evaluation report."""

    results: list[RagEvalResult]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    aggregate_score: float


class RegressionBaseline(BaseModel):
    """Stored baseline used by the regression harness."""

    average_judge_score: float
    average_rouge_l: float
    hallucination_rate: float
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ShadowEvaluationRecord(BaseModel):
    """One asynchronous shadow-eval log record."""

    request_id: str
    prompt: str
    response: str
    path: str
    method: str
    status_code: int
    sampled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rouge_l: float | None = None
    bertscore_f1: float | None = None
    judge_score: float | None = None
    error_class: ErrorClass | None = None
