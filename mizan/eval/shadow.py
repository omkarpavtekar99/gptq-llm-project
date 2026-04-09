"""Online shadow evaluation middleware for Phase 3."""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import uuid
from pathlib import Path

import mlflow
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import Settings
from mizan.eval.engine import EvalEngine
from mizan.eval.models import GoldenSetBundle, GoldenSetEntry, ModelExecutionConfig, ShadowEvaluationRecord
from mizan.logging_setup import get_logger

LOGGER = get_logger(__name__)


class ShadowEvaluationMiddleware(BaseHTTPMiddleware):
    """Capture a sample of live requests and score them asynchronously."""

    def __init__(self, app: object, settings: Settings, engine: EvalEngine) -> None:
        """Initialize the middleware."""

        super().__init__(app)
        self._settings = settings
        self._engine = engine
        self._lock = threading.Lock()
        self._ensure_database()

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Capture sampled requests and evaluate them in the background."""

        body = await request.body()
        response = await call_next(request)
        if random.random() > self._settings.eval.shadow_sample_rate:
            return response
        prompt = self._extract_prompt(body)
        response_text = getattr(response, "body", b"").decode("utf-8", errors="ignore")
        if not prompt or not response_text:
            return response
        worker = threading.Thread(
            target=self._score_shadow_request,
            args=(request, prompt, response_text, response.status_code),
            daemon=True,
        )
        worker.start()
        return response

    def _score_shadow_request(
        self, request: Request, prompt: str, response_text: str, status_code: int
    ) -> None:
        """Evaluate a shadow sample and persist it to MLflow and SQLite."""

        entry = GoldenSetEntry(
            id=str(uuid.uuid4()),
            prompt=prompt,
            expected_output=response_text,
            tags={"source": "shadow", "path": request.url.path},
            version="shadow",
        )
        report = self._engine.run_eval(
            GoldenSetBundle(name="shadow", version="shadow", entries=[entry]),
            ModelExecutionConfig(
                model_name=self._settings.vllm.model_name,
                temperature=0.0,
                max_tokens=self._settings.vllm.eval_max_tokens,
                timeout_sec=self._settings.eval.judge_timeout_sec,
            ),
        )
        result = report.results[0]
        record = ShadowEvaluationRecord(
            request_id=entry.id,
            prompt=prompt,
            response=response_text,
            path=request.url.path,
            method=request.method,
            status_code=status_code,
            rouge_l=result.rouge_l,
            bertscore_f1=result.bertscore_f1,
            judge_score=float(result.judge.score),
            error_class=result.error_class,
        )
        with self._lock:
            self._write_record(record)
        mlflow.log_dict(record.model_dump(mode="json"), f"shadow/{record.request_id}.json")

    def _ensure_database(self) -> None:
        """Create the local shadow log table when needed."""

        path = self._settings.paths.shadow_db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._settings.eval.shadow_table_name} (
                    request_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    path TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    sampled_at TEXT NOT NULL,
                    rouge_l REAL,
                    bertscore_f1 REAL,
                    judge_score REAL,
                    error_class TEXT
                )
                """
            )

    def _write_record(self, record: ShadowEvaluationRecord) -> None:
        """Append one shadow evaluation record to SQLite."""

        with sqlite3.connect(self._settings.paths.shadow_db_path) as connection:
            connection.execute(
                f"""
                INSERT OR REPLACE INTO {self._settings.eval.shadow_table_name}
                (request_id, prompt, response, path, method, status_code, sampled_at, rouge_l, bertscore_f1, judge_score, error_class)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.prompt,
                    record.response,
                    record.path,
                    record.method,
                    record.status_code,
                    record.sampled_at.isoformat(),
                    record.rouge_l,
                    record.bertscore_f1,
                    record.judge_score,
                    None if record.error_class is None else record.error_class.value,
                ),
            )

    @staticmethod
    def _extract_prompt(body: bytes) -> str | None:
        """Extract a prompt or latest user message from a JSON request body."""

        if not body:
            return None
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            return payload["prompt"]
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    return str(message.get("content", "")).strip() or None
        return None
