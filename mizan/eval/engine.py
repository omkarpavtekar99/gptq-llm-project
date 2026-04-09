"""Core evaluation engine for Phase 3."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from statistics import mean

import httpx
import mlflow
from rouge_score import rouge_scorer

from config.settings import Settings
from mizan.eval.models import (
    ErrorClass,
    EvalReport,
    EvalResult,
    GoldenSetBundle,
    GoldenSetEntry,
    JudgeResult,
    ModelExecutionConfig,
)
from mizan.logging_setup import get_logger

LOGGER = get_logger(__name__)

JUDGE_PROMPT = """You are an evaluation judge for English LLM responses.
Score the assistant response from 1 to 5 for accuracy, relevance, and completeness.
Return strict JSON with keys: score, reason.
Prompt: {prompt}
Expected output: {expected}
Actual output: {actual}
"""


class EvalEngine:
    """Run golden-set evaluation against the local model server."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the engine."""

        self._settings = settings
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def run_eval(self, golden_set: GoldenSetBundle, model_config: ModelExecutionConfig) -> EvalReport:
        """Run the full eval loop for one golden set."""

        results: list[EvalResult] = []
        mlflow.set_tracking_uri(self._settings.mlflow.tracking_uri)
        mlflow.set_experiment(self._settings.mlflow.experiment_name)
        with mlflow.start_run(run_name=f"eval:{golden_set.name}:{golden_set.version}", nested=True):
            mlflow.log_params(
                {
                    "golden_set_name": golden_set.name,
                    "golden_set_version": golden_set.version,
                    "model_name": model_config.model_name,
                    "temperature": model_config.temperature,
                    "max_tokens": model_config.max_tokens,
                }
            )
            for entry in golden_set.entries:
                results.append(self.evaluate_entry(entry, model_config))
            report = EvalReport(
                golden_set_name=golden_set.name,
                golden_set_version=golden_set.version,
                execution_config=model_config,
                results=results,
                average_rouge_l=round(mean(item.rouge_l for item in results), 4),
                average_bertscore_f1=round(mean(item.bertscore_f1 for item in results), 4),
                average_judge_score=round(mean(item.judge.score for item in results), 4),
                error_counts=dict(Counter(item.error_class.value for item in results if item.error_class)),
            )
            self._log_report(report)
            return report

    def evaluate_entry(self, entry: GoldenSetEntry, model_config: ModelExecutionConfig) -> EvalResult:
        """Evaluate one golden-set example."""

        started = time.perf_counter()
        response_text = self._generate_response(entry.prompt, model_config)
        response_latency_ms = round((time.perf_counter() - started) * 1000, 4)
        rouge_l = round(
            self._scorer.score(entry.expected_output, response_text)["rougeL"].fmeasure, 4
        )
        bertscore_f1 = self._calculate_bertscore(response_text, entry.expected_output)
        judge = self._judge_response(entry, response_text)
        error_class = self.classify_error(response_text, rouge_l, judge.score)
        return EvalResult(
            entry_id=entry.id,
            prompt=entry.prompt,
            expected_output=entry.expected_output,
            actual_output=response_text,
            rouge_l=rouge_l,
            bertscore_f1=bertscore_f1,
            judge=judge,
            error_class=error_class,
            response_latency_ms=response_latency_ms,
        )

    def classify_error(self, response_text: str, rouge_l: float, judge_score: int) -> ErrorClass | None:
        """Assign the shared error taxonomy."""

        lowered = response_text.lower()
        if any(token in lowered for token in ["i can't", "i cannot", "i'm sorry", "refuse"]):
            return ErrorClass.refusal
        if judge_score <= 2 and rouge_l < 0.1:
            return ErrorClass.hallucination
        if judge_score <= 3 or rouge_l < self._settings.thresholds.rouge_min:
            return ErrorClass.low_quality
        if not response_text.strip():
            return ErrorClass.format_error
        return None

    def _generate_response(self, prompt: str, model_config: ModelExecutionConfig) -> str:
        """Generate one response via the local OpenAI-compatible vLLM endpoint."""

        payload = {
            "model": model_config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
        }
        try:
            response = httpx.post(
                f"{self._settings.vllm.base_url}/chat/completions",
                json=payload,
                timeout=model_config.timeout_sec,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError("Model generation timed out.") from exc
        body = response.json()
        return str(body["choices"][0]["message"]["content"]).strip()

    def _calculate_bertscore(self, prediction: str, reference: str) -> float:
        """Compute BERTScore F1 with the configured lightweight encoder."""

        try:
            from bert_score import score
        except ImportError as exc:
            raise RuntimeError("bert-score is not installed. Run pip install -e '.[dev]'.") from exc
        _, _, f1 = score(
            [prediction],
            [reference],
            model_type=self._settings.eval.bertscore_model_type,
            verbose=False,
            lang="en",
        )
        return round(float(f1.mean().item()), 4)

    def _judge_response(self, entry: GoldenSetEntry, actual_output: str) -> JudgeResult:
        """Score one response with Qwen as an LLM judge."""

        judge_config = ModelExecutionConfig(
            model_name=self._settings.eval.judge_model_name,
            temperature=self._settings.eval.judge_temperature,
            max_tokens=self._settings.eval.judge_max_tokens,
            timeout_sec=self._settings.eval.judge_timeout_sec,
        )
        started = time.perf_counter()
        raw = self._generate_response(
            JUDGE_PROMPT.format(
                prompt=entry.prompt,
                expected=entry.expected_output,
                actual=actual_output,
            ),
            judge_config,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 4)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        try:
            payload = json.loads(match.group(0) if match else raw)
        except json.JSONDecodeError:
            payload = {"score": 1, "reason": raw[:240] or "Judge output was not valid JSON."}
        score_value = int(max(1, min(5, int(payload.get("score", 1)))))
        return JudgeResult(score=score_value, reason=str(payload.get("reason", "")), latency_ms=latency_ms)

    @staticmethod
    def _log_report(report: EvalReport) -> None:
        """Log aggregate eval metrics to MLflow."""

        mlflow.log_metrics(
            {
                "average_rouge_l": report.average_rouge_l,
                "average_bertscore_f1": report.average_bertscore_f1,
                "average_judge_score": report.average_judge_score,
            }
        )
        if report.error_counts:
            mlflow.log_dict(report.error_counts, "error_counts.json")
