from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from statistics import mean
from typing import cast

import typer

from meeting_pipeline.config import get_settings
from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.pgvector_search import ConnectionProtocol, PgVectorSearcher
from meeting_pipeline.embeddings.embedder import Embedder
from meeting_pipeline.embeddings.ollama_client import OllamaClient
from meeting_pipeline.rag.answer_generator import AnswerGenerator
from meeting_pipeline.rag.query_rewriter import QueryRewriter
from meeting_pipeline.rag.retriever import Retriever
from meeting_pipeline.timing import elapsed_ms, now

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Benchmark local RAG question latency and cache behavior.")


def _extract_timing_map(metadata: dict[str, object]) -> dict[str, float]:
    raw = metadata.get("timings_ms")
    if not isinstance(raw, dict):
        return {}

    parsed: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            parsed[key] = float(value)
    return parsed


def _extract_cache_map(metadata: dict[str, object]) -> dict[str, bool]:
    raw = metadata.get("cache")
    if not isinstance(raw, dict):
        return {}

    parsed: dict[str, bool] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, bool):
            parsed[key] = value
    return parsed


def _average_timings(timing_samples: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = {key for sample in timing_samples for key in sample.keys() if isinstance(key, str)}
    return {
        key: round(mean(sample.get(key, 0.0) for sample in timing_samples), 3)
        for key in sorted(keys)
    }


def _cache_hit_counts(cache_samples: Sequence[dict[str, bool]]) -> dict[str, int]:
    keys = {key for sample in cache_samples for key in sample.keys() if isinstance(key, str)}
    return {
        key: sum(1 for sample in cache_samples if sample.get(key, False)) for key in sorted(keys)
    }


def run_benchmark(
    *,
    meeting_id: str,
    questions: list[str],
    runs: int,
    top_k: int | None = None,
    fast_mode: bool = False,
    use_cache: bool = True,
) -> dict[str, object]:
    if runs <= 0:
        raise ValueError("runs must be a positive integer")

    normalized_questions = [
        " ".join(question.split()) for question in questions if question.strip()
    ]
    if not normalized_questions:
        raise ValueError("questions must contain at least one non-empty value")

    normalized_meeting_id = meeting_id.strip()
    if not normalized_meeting_id:
        raise ValueError("meeting_id must be a non-empty string")

    settings = get_settings()
    shared_client = OllamaClient.from_settings(settings)
    query_rewriter = QueryRewriter(client=shared_client, settings=settings)
    embedder = Embedder(client=shared_client, settings=settings)
    answer_generator = AnswerGenerator(client=shared_client, settings=settings)

    timing_samples: list[dict[str, float]] = []
    cache_samples: list[dict[str, bool]] = []

    with connection_scope(application_name="meeting_pipeline:benchmark_rag") as connection:
        searcher = PgVectorSearcher(cast(ConnectionProtocol, connection))
        retriever = Retriever(
            searcher=searcher,
            query_rewriter=query_rewriter,
            embedder=embedder,
            settings=settings,
        )

        for run_index in range(1, runs + 1):
            for question in normalized_questions:
                request_started_at = now()
                bundle = retriever.retrieve(
                    meeting_id=normalized_meeting_id,
                    user_query=question,
                    top_k=top_k,
                    use_cache=use_cache,
                    fast_mode=fast_mode,
                )
                answer = answer_generator.generate(
                    user_question=question,
                    meeting_id=normalized_meeting_id,
                    rewritten_query=bundle.rewritten_query,
                    retrieved_evidence=bundle.results,
                    retrieval_mode=bundle.retrieval_mode,
                    use_cache=use_cache,
                    fast_mode=fast_mode,
                )

                timings = _extract_timing_map(bundle.service_metadata)
                timings.update(_extract_timing_map(answer.service_metadata))
                timings["total_request"] = elapsed_ms(request_started_at)
                timing_samples.append(timings)

                cache_map = _extract_cache_map(bundle.service_metadata)
                cache_map.update(_extract_cache_map(answer.service_metadata))
                cache_samples.append(cache_map)

                LOGGER.info(
                    "benchmark_run=%d question=%s total_ms=%.2f",
                    run_index,
                    question,
                    timings.get("total_request", 0.0),
                )

    averages = _average_timings(timing_samples)
    cache_hits = _cache_hit_counts(cache_samples)
    sample_count = len(timing_samples)

    return {
        "meeting_id": normalized_meeting_id,
        "question_count": len(normalized_questions),
        "runs": runs,
        "total_samples": sample_count,
        "fast_mode": fast_mode,
        "use_cache": use_cache,
        "average_latency_ms": averages.get("total_request", 0.0),
        "average_stage_timings_ms": averages,
        "cache_hit_counts": cache_hits,
        "cache_hit_rates": {
            key: round((value / sample_count) if sample_count else 0.0, 3)
            for key, value in cache_hits.items()
        },
    }


@app.command()
def main(
    meeting_id: str = typer.Option(..., "--meeting-id"),
    question: list[str] = typer.Option(..., "--question"),
    runs: int = typer.Option(3, "--runs"),
    top_k: int | None = typer.Option(None, "--top-k"),
    fast_mode: bool = typer.Option(False, "--fast-mode"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        payload = run_benchmark(
            meeting_id=meeting_id,
            questions=question,
            runs=runs,
            top_k=top_k,
            fast_mode=fast_mode,
            use_cache=not no_cache,
        )
    except Exception as exc:
        LOGGER.error("benchmark run failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
