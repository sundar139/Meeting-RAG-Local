from __future__ import annotations

import json
import logging
from dataclasses import asdict

import typer

from meeting_pipeline.config import get_settings
from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.pgvector_search import PgVectorSearcher
from meeting_pipeline.embeddings.embedder import Embedder
from meeting_pipeline.embeddings.ollama_client import OllamaClient
from meeting_pipeline.rag.answer_generator import AnswerGenerator
from meeting_pipeline.rag.query_rewriter import QueryRewriter
from meeting_pipeline.rag.retriever import Retriever
from meeting_pipeline.timing import elapsed_ms, now

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Smoke-test retrieval and grounded answer generation.")


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


def _latency_summary(timings: dict[str, float]) -> str:
    label_candidates = [
        (("query_rewrite",), "rewrite"),
        (("query_embedding", "embedding_query_prep"), "embed"),
        (("postgres_retrieval", "retrieval"), "retrieve"),
        (("answer_generation",), "answer"),
        (("total_request", "retrieval_total"), "total"),
    ]
    parts: list[str] = []
    for candidates, short in label_candidates:
        for key in candidates:
            if key in timings:
                parts.append(f"{short} {timings[key]:.1f} ms")
                break
    return " | ".join(parts)


def _cache_summary(cache: dict[str, bool]) -> str:
    labels = [
        ("query_rewrite", "rewrite"),
        ("query_embedding", "embed"),
        ("retrieval_bundle", "retrieve"),
        ("answer_generation", "answer"),
    ]
    return " | ".join(
        f"{short} {'hit' if cache.get(key, False) else 'miss'}" for key, short in labels
    )


def run_smoke(
    *,
    meeting_id: str,
    question: str,
    top_k: int = 5,
    context: list[str] | None = None,
    debug: bool = False,
    preview_evidence: bool = False,
    fast_mode: bool = False,
    use_cache: bool = True,
) -> dict[str, object]:
    settings = get_settings()
    shared_client = OllamaClient.from_settings(settings)
    query_rewriter = QueryRewriter(client=shared_client, settings=settings)
    embedder = Embedder(client=shared_client, settings=settings)
    answer_generator = AnswerGenerator(client=shared_client, settings=settings)

    request_started_at = now()
    with connection_scope(application_name="meeting_pipeline:smoke_rag") as connection:
        searcher = PgVectorSearcher(connection)
        retriever = Retriever(
            searcher=searcher,
            query_rewriter=query_rewriter,
            embedder=embedder,
            settings=settings,
        )
        bundle = retriever.retrieve(
            meeting_id=meeting_id,
            user_query=question,
            conversation_context=context,
            top_k=top_k,
            use_cache=use_cache,
            fast_mode=fast_mode,
        )

    if debug and preview_evidence:
        preview_payload = {
            "meeting_id": bundle.meeting_id,
            "retrieval_mode": bundle.retrieval_mode,
            "top_k_used": bundle.top_k_used,
            "used_cached_context": bundle.used_cached_context,
            "rewritten_query": bundle.rewritten_query,
            "evidence_preview": [asdict(item) for item in bundle.results],
        }
        LOGGER.info("retrieval_preview_before_answer=%s", json.dumps(preview_payload))

    answer = answer_generator.generate(
        user_question=question,
        meeting_id=meeting_id,
        rewritten_query=bundle.rewritten_query,
        retrieved_evidence=bundle.results,
        conversation_context=context,
        retrieval_mode=bundle.retrieval_mode,
        use_cache=use_cache,
        fast_mode=fast_mode,
    )

    timings = _extract_timing_map(bundle.service_metadata)
    timings.update(_extract_timing_map(answer.service_metadata))
    timings["total_request"] = elapsed_ms(request_started_at)
    normalized_cache: dict[str, bool] = _extract_cache_map(bundle.service_metadata)
    normalized_cache.update(_extract_cache_map(answer.service_metadata))

    if debug:
        LOGGER.info("latency_summary=%s", _latency_summary(timings))
        LOGGER.info("cache_summary=%s", _cache_summary(normalized_cache))

    return {
        "meeting_id": bundle.meeting_id,
        "user_query": bundle.user_query,
        "rewritten_query": bundle.rewritten_query,
        "retrieval_mode": bundle.retrieval_mode,
        "top_k_used": bundle.top_k_used,
        "used_cached_context": bundle.used_cached_context,
        "evidence_count": len(bundle.results),
        "evidence": [asdict(item) for item in bundle.results],
        "service_metadata": {
            "timings_ms": timings,
            "cache": normalized_cache,
            "fast_mode": fast_mode,
        },
        "answer": {
            "sections": answer.sections,
            "insufficient_context": answer.insufficient_context,
            "confidence_tier": answer.confidence_tier,
            "raw_answer": answer.raw_answer,
            "service_metadata": answer.service_metadata,
        },
    }


@app.command()
def main(
    meeting_id: str = typer.Option(..., "--meeting-id"),
    question: str = typer.Option(..., "--question"),
    top_k: int = typer.Option(5, "--top-k"),
    context: list[str] | None = typer.Option(None, "--context"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug diagnostics."),
    preview_evidence: bool = typer.Option(
        False,
        "--preview-evidence",
        help="When debugging, print retrieved evidence before answer synthesis.",
    ),
    fast_mode: bool = typer.Option(False, "--fast-mode", help="Enable fast-mode latency path."),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Bypass in-memory caches for this run.",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        payload = run_smoke(
            meeting_id=meeting_id,
            question=question,
            top_k=top_k,
            context=context,
            debug=debug or preview_evidence,
            preview_evidence=preview_evidence,
            fast_mode=fast_mode,
            use_cache=not no_cache,
        )
    except Exception as exc:
        LOGGER.error("smoke retrieval/answer run failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
