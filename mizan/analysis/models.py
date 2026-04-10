"""Shared data models for Phase 4 drift detection and RCA."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class DriftBaseline(BaseModel):
    """Persisted baseline embedding statistics."""

    centroid: list[float]
    covariance: list[list[float]]
    judge_score_baseline: float
    output_histogram: list[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DriftReport(BaseModel):
    """Summary of drift status for one eval run."""

    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    js_divergence: float
    cosine_drift: float
    rolling_judge_score: float
    per_tag_scores: dict[str, float]
    baseline_deltas: dict[str, float]
    alert_triggered: bool
    alert_reasons: list[str]

    def to_markdown(self) -> str:
        """Render the drift report as Markdown."""

        lines = [
            "# Phase 4 Drift Report",
            "",
            f"- run_id: `{self.run_id}`",
            f"- js_divergence: `{self.js_divergence:.4f}`",
            f"- cosine_drift: `{self.cosine_drift:.4f}`",
            f"- rolling_judge_score: `{self.rolling_judge_score:.4f}`",
            f"- alert_triggered: `{self.alert_triggered}`",
            "",
            "## Per-tag Scores",
        ]
        for tag, score in sorted(self.per_tag_scores.items()):
            lines.append(f"- {tag}: `{score:.4f}`")
        lines.extend(["", "## Baseline Deltas"])
        for metric, delta in sorted(self.baseline_deltas.items()):
            lines.append(f"- {metric}: `{delta:.4f}`")
        if self.alert_reasons:
            lines.extend(["", "## Alert Reasons"])
            for reason in self.alert_reasons:
                lines.append(f"- {reason}")
        lines.append("")
        return "\n".join(lines)


class RcaReport(BaseModel):
    """Outcome of one root-cause comparison."""

    winner: str
    score_delta: float
    latency_delta: float
    recommendation: str
    metadata: dict[str, Any] = Field(default_factory=dict)
