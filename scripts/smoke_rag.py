from __future__ import annotations

import json
import logging
from dataclasses import asdict

import typer

from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.pgvector_search import PgVectorSearcher
from meeting_pipeline.rag.answer_generator import AnswerGenerator
from meeting_pipeline.rag.retriever import Retriever

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Smoke-test retrieval and grounded answer generation.")


def run_smoke(
    *,
    meeting_id: str,
    question: str,
    top_k: int = 5,
    context: list[str] | None = None,
) -> dict[str, object]:
    with connection_scope(application_name="meeting_pipeline:smoke_rag") as connection:
        searcher = PgVectorSearcher(connection)
        retriever = Retriever(searcher=searcher)
        bundle = retriever.retrieve(
            meeting_id=meeting_id,
            user_query=question,
            conversation_context=context,
            top_k=top_k,
        )

    answer_generator = AnswerGenerator()
    answer = answer_generator.generate(
        user_question=question,
        meeting_id=meeting_id,
        rewritten_query=bundle.rewritten_query,
        retrieved_evidence=bundle.results,
        conversation_context=context,
    )

    return {
        "meeting_id": bundle.meeting_id,
        "user_query": bundle.user_query,
        "rewritten_query": bundle.rewritten_query,
        "top_k_used": bundle.top_k_used,
        "evidence_count": len(bundle.results),
        "evidence": [asdict(item) for item in bundle.results],
        "answer": {
            "sections": answer.sections,
            "insufficient_context": answer.insufficient_context,
            "raw_answer": answer.raw_answer,
        },
    }


@app.command()
def main(
    meeting_id: str = typer.Option(..., "--meeting-id"),
    question: str = typer.Option(..., "--question"),
    top_k: int = typer.Option(5, "--top-k"),
    context: list[str] | None = typer.Option(None, "--context"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        payload = run_smoke(
            meeting_id=meeting_id,
            question=question,
            top_k=top_k,
            context=context,
        )
    except Exception as exc:
        LOGGER.error("smoke retrieval/answer run failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
