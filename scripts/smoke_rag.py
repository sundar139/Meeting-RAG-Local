from __future__ import annotations

import json
import logging
from dataclasses import asdict

import typer

from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.pgvector_search import PgVectorSearcher
from meeting_pipeline.rag.answer_generator import AnswerGenerator
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


def _latency_summary(timings: dict[str, float]) -> str:
    labels = [
        ("query_rewrite", "rewrite"),
        ("embedding_query_prep", "embed"),
        ("retrieval", "retrieve"),
        ("answer_generation", "answer"),
        ("total_request", "total"),
    ]
    return " | ".join(f"{short} {timings[key]:.1f} ms" for key, short in labels if key in timings)


def run_smoke(
    *,
    meeting_id: str,
    question: str,
    top_k: int = 5,
    context: list[str] | None = None,
    debug: bool = False,
    preview_evidence: bool = False,
) -> dict[str, object]:
    request_started_at = now()
    with connection_scope(application_name="meeting_pipeline:smoke_rag") as connection:
        searcher = PgVectorSearcher(connection)
        retriever = Retriever(searcher=searcher)
        bundle = retriever.retrieve(
            meeting_id=meeting_id,
            user_query=question,
            conversation_context=context,
            top_k=top_k,
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

    answer_generator = AnswerGenerator()
    answer = answer_generator.generate(
        user_question=question,
        meeting_id=meeting_id,
        rewritten_query=bundle.rewritten_query,
        retrieved_evidence=bundle.results,
        conversation_context=context,
        retrieval_mode=bundle.retrieval_mode,
    )

    timings = _extract_timing_map(bundle.service_metadata)
    timings.update(_extract_timing_map(answer.service_metadata))
    timings["total_request"] = elapsed_ms(request_started_at)

    if debug:
        LOGGER.info("latency_summary=%s", _latency_summary(timings))

    return {
        "meeting_id": bundle.meeting_id,
        "user_query": bundle.user_query,
        "rewritten_query": bundle.rewritten_query,
        "retrieval_mode": bundle.retrieval_mode,
        "top_k_used": bundle.top_k_used,
        "used_cached_context": bundle.used_cached_context,
        "evidence_count": len(bundle.results),
        "evidence": [asdict(item) for item in bundle.results],
        "service_metadata": {"timings_ms": timings},
        "answer": {
            "sections": answer.sections,
            "insufficient_context": answer.insufficient_context,
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
        )
    except Exception as exc:
        LOGGER.error("smoke retrieval/answer run failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
