"""RAG evaluation helpers for Phase 3."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import mlflow

from config.settings import Settings
from mizan.eval.models import RagEvalReport, RagEvalResult

PUBLIC_DOMAIN_DOCS: list[tuple[str, str]] = [
    ("alice_01", "Alice was beginning to get very tired of sitting by her sister on the bank."),
    ("alice_02", "The rabbit actually took a watch out of its waistcoat-pocket and looked at it."),
    ("pride_01", "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."),
    ("pride_02", "Elizabeth listened with delight to the lively, playful disposition of her friend."),
    ("frankenstein_01", "I beheld the accomplishment of my toils with an anxiety that almost amounted to agony."),
    ("frankenstein_02", "Learn from me, if not by my precepts, at least by my example."),
    ("sherlock_01", "You see, but you do not observe."),
    ("sherlock_02", "There is nothing more deceptive than an obvious fact."),
    ("moby_01", "Call me Ishmael."),
    ("moby_02", "Whenever I find myself growing grim about the mouth, I account it high time to get to sea."),
    ("time_machine_01", "The Time Traveller was expounding a recondite matter to us."),
    ("time_machine_02", "We all know that Time is only a kind of Space."),
    ("aesop_01", "A fox saw a crow fly off with a piece of cheese in its beak."),
    ("aesop_02", "The wind and the sun disputed which was the stronger."),
    ("beowulf_01", "Then a mighty warrior went to visit the great hall Heorot."),
    ("beowulf_02", "The monster Grendel prowled the night in anger."),
    ("iliad_01", "Sing, goddess, the anger of Achilles son of Peleus."),
    ("iliad_02", "The will of Zeus was moving toward its end."),
    ("odyssey_01", "Tell me, Muse, of the man of many ways."),
    ("odyssey_02", "He saw the cities of many people and learned their minds."),
]


class RagEvaluator:
    """Build a small ChromaDB index and score RAG quality with RAGAS."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the evaluator."""

        self._settings = settings

    def ensure_sample_corpus(self) -> list[Path]:
        """Materialize the public-domain sample corpus to disk."""

        self._settings.paths.rag_docs_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []
        for doc_id, content in PUBLIC_DOMAIN_DOCS:
            path = self._settings.paths.rag_docs_dir / f"{doc_id}.txt"
            if not path.exists():
                path.write_text(content, encoding="utf-8")
            created.append(path)
        return created

    def run_eval(self, questions: list[str], answers: list[str], ground_truths: list[str]) -> RagEvalReport:
        """Build the local vector store and evaluate the supplied question set."""

        contexts = self._retrieve_contexts(questions)
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("ragas and its dataset dependencies are not installed.") from exc
        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        score_frame = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        ).to_pandas()
        results = [
            RagEvalResult(
                question=questions[index],
                answer=answers[index],
                contexts=contexts[index],
                ground_truth=ground_truths[index],
                faithfulness=round(float(score_frame.iloc[index]["faithfulness"]), 4),
                answer_relevancy=round(float(score_frame.iloc[index]["answer_relevancy"]), 4),
                context_precision=round(float(score_frame.iloc[index]["context_precision"]), 4),
                context_recall=round(float(score_frame.iloc[index]["context_recall"]), 4),
            )
            for index in range(len(questions))
        ]
        report = RagEvalReport(
            results=results,
            faithfulness=round(mean(item.faithfulness for item in results), 4),
            answer_relevancy=round(mean(item.answer_relevancy for item in results), 4),
            context_precision=round(mean(item.context_precision for item in results), 4),
            context_recall=round(mean(item.context_recall for item in results), 4),
            aggregate_score=round(
                mean(
                    [
                        mean(
                            [
                                item.faithfulness,
                                item.answer_relevancy,
                                item.context_precision,
                                item.context_recall,
                            ]
                        )
                        for item in results
                    ]
                ),
                4,
            ),
        )
        mlflow.log_metrics(
            {
                "rag_faithfulness": report.faithfulness,
                "rag_answer_relevancy": report.answer_relevancy,
                "rag_context_precision": report.context_precision,
                "rag_context_recall": report.context_recall,
                "rag_aggregate_score": report.aggregate_score,
            }
        )
        return report

    def _retrieve_contexts(self, questions: list[str]) -> list[list[str]]:
        """Build the ChromaDB collection and return top-k contexts per question."""

        self.ensure_sample_corpus()
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("chromadb and sentence-transformers are not installed.") from exc
        documents = [path.read_text(encoding="utf-8") for path in sorted(self._settings.paths.rag_docs_dir.glob("*.txt"))]
        ids = [path.stem for path in sorted(self._settings.paths.rag_docs_dir.glob("*.txt"))]
        model = SentenceTransformer(self._settings.rag.embedding_model)
        embeddings = model.encode(documents).tolist()
        client = chromadb.PersistentClient(path=str(self._settings.rag.chroma_persist_dir))
        if self._settings.eval.rag_rebuild_index:
            try:
                client.delete_collection(self._settings.eval.rag_collection_name)
            except Exception:
                pass
        collection = client.get_or_create_collection(self._settings.eval.rag_collection_name)
        if collection.count() == 0 or self._settings.eval.rag_rebuild_index:
            collection.upsert(ids=ids, documents=documents, embeddings=embeddings)
        question_embeddings = model.encode(questions).tolist()
        query = collection.query(query_embeddings=question_embeddings, n_results=self._settings.eval.rag_top_k)
        return [[str(item) for item in docs] for docs in query["documents"]]
