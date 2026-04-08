from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.pgvector_search import PgVectorSearcher
from meeting_pipeline.eval.retrieval_eval import (
    RetrievalPredictionItem,
    RetrievedEvidence,
    evaluate_retrieval_benchmark,
    load_retrieval_benchmark,
    load_retrieval_predictions,
)
from meeting_pipeline.eval.transcript_eval import evaluate_transcript_files
from meeting_pipeline.rag.retriever import Retriever

app = typer.Typer(help="Evaluation pipeline entry point.")
LOGGER = logging.getLogger(__name__)


def _generate_live_retrieval_predictions(
    benchmark_path: Path,
    *,
    top_k: int,
) -> list[RetrievalPredictionItem]:
    benchmark_items = load_retrieval_benchmark(benchmark_path)

    predictions: list[RetrievalPredictionItem] = []
    with connection_scope(application_name="meeting_pipeline:run_eval_retrieval") as connection:
        retriever = Retriever(searcher=PgVectorSearcher(connection))

        for item in benchmark_items:
            bundle = retriever.retrieve(
                meeting_id=item.meeting_id,
                user_query=item.question,
                top_k=top_k,
            )
            predictions.append(
                RetrievalPredictionItem(
                    meeting_id=item.meeting_id,
                    question=item.question,
                    rewritten_query=bundle.rewritten_query,
                    retrieved=[
                        RetrievedEvidence(
                            chunk_id=chunk.chunk_id,
                            speaker_label=chunk.speaker_label,
                            content=chunk.content,
                            similarity=chunk.similarity,
                        )
                        for chunk in bundle.results
                    ],
                )
            )

    return predictions


def run_evaluation(
    *,
    transcript_reference_path: Path | None = None,
    transcript_prediction_path: Path | None = None,
    retrieval_benchmark_path: Path | None = None,
    retrieval_predictions_path: Path | None = None,
    retrieval_top_k: int = 5,
    live_retrieval: bool = False,
    include_item_details: bool = False,
) -> dict[str, object]:
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be a positive integer")

    summary: dict[str, object] = {}

    has_transcript_args = (
        transcript_reference_path is not None or transcript_prediction_path is not None
    )
    if has_transcript_args and (
        transcript_reference_path is None or transcript_prediction_path is None
    ):
        raise ValueError(
            "Both transcript_reference_path and transcript_prediction_path are required "
            "for transcript evaluation."
        )

    if transcript_reference_path is not None and transcript_prediction_path is not None:
        transcript_result = evaluate_transcript_files(
            reference_path=transcript_reference_path,
            prediction_path=transcript_prediction_path,
        )
        summary["transcript"] = transcript_result.to_summary()

    if retrieval_benchmark_path is not None:
        benchmark_items = load_retrieval_benchmark(retrieval_benchmark_path)

        prediction_items: list[RetrievalPredictionItem]
        if retrieval_predictions_path is not None:
            prediction_items = load_retrieval_predictions(retrieval_predictions_path)
        elif live_retrieval:
            prediction_items = _generate_live_retrieval_predictions(
                retrieval_benchmark_path,
                top_k=retrieval_top_k,
            )
        else:
            raise ValueError(
                "Retrieval evaluation requires either retrieval_predictions_path "
                "or live_retrieval=True."
            )

        retrieval_result = evaluate_retrieval_benchmark(
            benchmark_items,
            prediction_items,
            top_k=retrieval_top_k,
        )
        summary["retrieval"] = retrieval_result.to_summary(include_items=include_item_details)

    if not summary:
        raise ValueError(
            "No evaluation inputs provided. Provide transcript files and/or retrieval "
            "benchmark inputs."
        )

    return summary


@app.command()
def main(
    transcript_reference_path: Path | None = typer.Option(None, "--transcript-reference-path"),
    transcript_prediction_path: Path | None = typer.Option(None, "--transcript-prediction-path"),
    retrieval_benchmark_path: Path | None = typer.Option(None, "--retrieval-benchmark-path"),
    retrieval_predictions_path: Path | None = typer.Option(None, "--retrieval-predictions-path"),
    retrieval_top_k: int = typer.Option(5, "--retrieval-top-k"),
    live_retrieval: bool = typer.Option(False, "--live-retrieval"),
    output_path: Path | None = typer.Option(None, "--output-path"),
    include_item_details: bool = typer.Option(False, "--include-item-details"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        summary = run_evaluation(
            transcript_reference_path=transcript_reference_path,
            transcript_prediction_path=transcript_prediction_path,
            retrieval_benchmark_path=retrieval_benchmark_path,
            retrieval_predictions_path=retrieval_predictions_path,
            retrieval_top_k=retrieval_top_k,
            live_retrieval=live_retrieval,
            include_item_details=include_item_details,
        )
    except Exception as exc:
        LOGGER.error("evaluation failed: %s", exc)
        raise typer.Exit(code=1) from exc

    output = json.dumps(summary, indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    typer.echo(output)


if __name__ == "__main__":
    app()
