from __future__ import annotations

import pytest

from meeting_pipeline.config import Settings
from meeting_pipeline.rag.models import GroundedAnswerResult, RetrievalBundle
from scripts.benchmark_rag import run_benchmark


class _FakeScope:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type
        _ = exc
        _ = tb
        return None


class FakeRetriever:
    def __init__(self, **_kwargs) -> None:
        self._seen: set[tuple[str, str, int | None, bool, bool]] = set()

    def retrieve(
        self,
        meeting_id: str,
        user_query: str,
        *,
        top_k: int | None = None,
        use_cache: bool = True,
        fast_mode: bool = False,
        **_kwargs,
    ) -> RetrievalBundle:
        key = (meeting_id, user_query, top_k, use_cache, fast_mode)
        hit = key in self._seen
        self._seen.add(key)
        return RetrievalBundle(
            meeting_id=meeting_id,
            user_query=user_query,
            rewritten_query=user_query,
            top_k_used=top_k or 4,
            results=[],
            retrieval_mode="default_factoid",
            service_metadata={
                "timings_ms": {
                    "query_rewrite": 1.0,
                    "query_embedding": 2.0,
                    "postgres_retrieval": 3.0,
                },
                "cache": {
                    "query_rewrite": hit,
                    "query_embedding": hit,
                    "retrieval_bundle": hit,
                },
            },
        )


class FakeAnswerGenerator:
    def __init__(self, **_kwargs) -> None:
        self._seen: set[tuple[str, str, bool, bool]] = set()

    def generate(
        self,
        *,
        meeting_id: str,
        rewritten_query: str,
        use_cache: bool = True,
        fast_mode: bool = False,
        **_kwargs,
    ) -> GroundedAnswerResult:
        key = (meeting_id, rewritten_query, use_cache, fast_mode)
        hit = key in self._seen
        self._seen.add(key)

        return GroundedAnswerResult(
            meeting_id=meeting_id,
            question=rewritten_query,
            rewritten_query=rewritten_query,
            sections={"Summary": "ok"},
            raw_answer="ok",
            insufficient_context=False,
            service_metadata={
                "timings_ms": {"answer_generation": 4.0},
                "cache": {"answer_generation": hit},
            },
        )


def test_run_benchmark_aggregates_stage_timings_and_cache_hits(monkeypatch) -> None:
    monkeypatch.setattr("scripts.benchmark_rag.get_settings", lambda: Settings.model_validate({}))
    monkeypatch.setattr("scripts.benchmark_rag.connection_scope", lambda **_kwargs: _FakeScope())
    monkeypatch.setattr(
        "scripts.benchmark_rag.OllamaClient.from_settings",
        lambda _settings: object(),
    )
    monkeypatch.setattr("scripts.benchmark_rag.QueryRewriter", lambda **_kwargs: object())
    monkeypatch.setattr("scripts.benchmark_rag.Embedder", lambda **_kwargs: object())
    monkeypatch.setattr("scripts.benchmark_rag.PgVectorSearcher", lambda _connection: object())
    monkeypatch.setattr("scripts.benchmark_rag.Retriever", lambda **kwargs: FakeRetriever(**kwargs))
    monkeypatch.setattr(
        "scripts.benchmark_rag.AnswerGenerator",
        lambda **kwargs: FakeAnswerGenerator(**kwargs),
    )

    summary = run_benchmark(
        meeting_id="m1",
        questions=["What happened?"],
        runs=2,
        top_k=4,
        fast_mode=True,
        use_cache=True,
    )

    assert summary["meeting_id"] == "m1"
    assert summary["total_samples"] == 2
    assert summary["fast_mode"] is True

    average_timings = summary["average_stage_timings_ms"]
    assert isinstance(average_timings, dict)
    assert average_timings["query_rewrite"] == 1.0
    assert average_timings["answer_generation"] == 4.0

    cache_hits = summary["cache_hit_counts"]
    assert isinstance(cache_hits, dict)
    assert cache_hits["retrieval_bundle"] == 1
    assert cache_hits["answer_generation"] == 1


def test_run_benchmark_requires_questions_and_positive_runs() -> None:
    with pytest.raises(ValueError, match="runs"):
        run_benchmark(meeting_id="m1", questions=["q"], runs=0)

    with pytest.raises(ValueError, match="questions"):
        run_benchmark(meeting_id="m1", questions=["   "], runs=1)
