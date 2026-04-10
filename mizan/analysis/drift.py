"""Drift detection for Phase 4."""

from __future__ import annotations

import pickle
from pathlib import Path
from statistics import mean
from uuid import uuid4

import mlflow
import numpy as np

from config.settings import Settings
from mizan.analysis.models import DriftBaseline, DriftReport
from mizan.eval.models import EvalReport
from mizan.logging_setup import get_logger

LOGGER = get_logger(__name__)


class DriftDetector:
    """Compute embedding, judge, and distribution drift against a baseline."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the detector."""

        self._settings = settings

    def recompute_baseline(self, report: EvalReport) -> DriftBaseline:
        """Generate and persist a new drift baseline from an eval report."""

        embeddings = self._embed_outputs([result.actual_output for result in report.results])
        baseline = DriftBaseline(
            centroid=np.mean(embeddings, axis=0).tolist(),
            covariance=np.cov(np.asarray(embeddings).T).tolist(),
            judge_score_baseline=report.average_judge_score,
            output_histogram=self._build_histogram(embeddings),
        )
        self._settings.paths.drift_baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with self._settings.paths.drift_baseline_path.open("wb") as handle:
            pickle.dump(baseline.model_dump(mode="json"), handle)
        return baseline

    def load_baseline(self) -> DriftBaseline:
        """Load the persisted drift baseline."""

        with self._settings.paths.drift_baseline_path.open("rb") as handle:
            payload = pickle.load(handle)
        return DriftBaseline.model_validate(payload)

    def detect(self, report: EvalReport, baseline: DriftBaseline | None = None) -> DriftReport:
        """Compare the current eval report against the saved baseline."""

        current_baseline = baseline or self.load_baseline()
        embeddings = self._embed_outputs([result.actual_output for result in report.results])
        centroid = np.asarray(current_baseline.centroid, dtype=np.float64)
        distances = [self._cosine_distance(np.asarray(vector), centroid) for vector in embeddings]
        current_histogram = self._build_histogram(embeddings)
        js_divergence = round(
            self._jensen_shannon_divergence(current_histogram, current_baseline.output_histogram), 4
        )
        cosine_drift = round(float(mean(distances)), 4)
        rolling_judge = round(report.average_judge_score, 4)
        per_tag_scores = self._per_tag_scores(report)
        deltas = {
            "js_divergence": js_divergence,
            "cosine_drift": cosine_drift,
            "judge_delta": round(abs(rolling_judge - current_baseline.judge_score_baseline), 4),
        }
        reasons = [
            reason
            for reason, value in deltas.items()
            if value > self._settings.thresholds.drift_alert_delta
        ]
        drift_report = DriftReport(
            run_id=str(uuid4()),
            js_divergence=js_divergence,
            cosine_drift=cosine_drift,
            rolling_judge_score=rolling_judge,
            per_tag_scores=per_tag_scores,
            baseline_deltas=deltas,
            alert_triggered=bool(reasons),
            alert_reasons=reasons,
        )
        self._log_report(drift_report)
        return drift_report

    def _embed_outputs(self, outputs: list[str]) -> list[list[float]]:
        """Embed generated outputs with the configured sentence-transformers model."""

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed.") from exc
        model = SentenceTransformer(self._settings.rag.embedding_model)
        return model.encode(outputs).tolist()

    @staticmethod
    def _build_histogram(embeddings: list[list[float]]) -> list[float]:
        """Project embeddings into a stable histogram for JS divergence."""

        flattened = np.asarray(embeddings, dtype=np.float64).flatten()
        histogram, _ = np.histogram(flattened, bins=20, range=(-1.0, 1.0), density=True)
        histogram = histogram / max(histogram.sum(), 1e-12)
        return histogram.tolist()

    @staticmethod
    def _cosine_distance(vector: np.ndarray, centroid: np.ndarray) -> float:
        """Compute cosine distance from one embedding to the baseline centroid."""

        denom = (np.linalg.norm(vector) * np.linalg.norm(centroid)) or 1e-12
        return float(1 - np.dot(vector, centroid) / denom)

    @staticmethod
    def _jensen_shannon_divergence(left: list[float], right: list[float]) -> float:
        """Compute JS divergence between two normalized histograms."""

        p = np.asarray(left, dtype=np.float64)
        q = np.asarray(right, dtype=np.float64)
        p = p / max(p.sum(), 1e-12)
        q = q / max(q.sum(), 1e-12)
        m = 0.5 * (p + q)
        kl_pm = np.sum(np.where(p > 0, p * np.log2(p / np.maximum(m, 1e-12)), 0.0))
        kl_qm = np.sum(np.where(q > 0, q * np.log2(q / np.maximum(m, 1e-12)), 0.0))
        return float(0.5 * (kl_pm + kl_qm))

    @staticmethod
    def _per_tag_scores(report: EvalReport) -> dict[str, float]:
        """Average judge scores grouped by flattened entry tags."""

        grouped: dict[str, list[int]] = {}
        for result in report.results:
            if not result.tags:
                grouped.setdefault("untagged", []).append(result.judge.score)
                continue
            for key, value in result.tags.items():
                grouped.setdefault(f"{key}:{value}", []).append(result.judge.score)
        return {tag: round(mean(scores), 4) for tag, scores in grouped.items()}

    @staticmethod
    def _log_report(report: DriftReport) -> None:
        """Log drift metrics and alert status to MLflow."""

        mlflow.log_metrics(
            {
                "drift_js_divergence": report.js_divergence,
                "drift_cosine_score": report.cosine_drift,
                "drift_rolling_judge_score": report.rolling_judge_score,
            }
        )
        mlflow.log_dict(report.model_dump(mode="json"), f"drift/{report.run_id}.json")
        if report.alert_triggered:
            LOGGER.warning("drift_alert_triggered", extra=report.model_dump(mode="json"))
