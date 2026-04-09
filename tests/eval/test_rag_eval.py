"""Tests for the RAG evaluator corpus bootstrap."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from mizan.eval.rag_eval import RagEvaluator


def test_ensure_sample_corpus_writes_documents(tmp_path: Path) -> None:
    """The evaluator should materialize its sample corpus on disk."""

    settings = Settings()
    settings.paths.rag_docs_dir = tmp_path
    evaluator = RagEvaluator(settings)

    created = evaluator.ensure_sample_corpus()

    assert len(created) == 20
    assert created[0].exists()
