from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from meeting_pipeline.eval.metrics import rate, safe_divide


@dataclass(frozen=True)
class RetrievalBenchmarkItem:
    meeting_id: str
    question: str
    expected_chunk_ids: list[int]
    expected_speaker_labels: list[str]
    expected_hints: list[str]


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: int | None
    speaker_label: str
    content: str
    similarity: float | None


@dataclass(frozen=True)
class RetrievalPredictionItem:
    meeting_id: str
    question: str
    rewritten_query: str
    retrieved: list[RetrievedEvidence]


@dataclass(frozen=True)
class RetrievalItemResult:
    meeting_id: str
    question: str
    retrieved_count: int
    chunk_recall_at_k: float
    chunk_hit: bool
    speaker_hit: bool
    hint_hit: bool
    evidence_hit: bool


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    query_count: int
    top_k: int
    recall_at_k: float
    evidence_hit_rate: float
    empty_retrieval_rate: float
    chunk_hit_rate: float
    speaker_hit_rate: float
    hint_hit_rate: float
    missing_predictions: int
    misses: list[dict[str, str]]
    item_results: list[RetrievalItemResult]

    def to_summary(self, *, include_items: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "query_count": self.query_count,
            "top_k": self.top_k,
            "recall_at_k": self.recall_at_k,
            "evidence_hit_rate": self.evidence_hit_rate,
            "empty_retrieval_rate": self.empty_retrieval_rate,
            "chunk_hit_rate": self.chunk_hit_rate,
            "speaker_hit_rate": self.speaker_hit_rate,
            "hint_hit_rate": self.hint_hit_rate,
            "missing_predictions": self.missing_predictions,
            "misses": self.misses,
        }
        if include_items:
            payload["items"] = [asdict(item) for item in self.item_results]
        return payload


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Retrieval evaluation file does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Retrieval evaluation file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Retrieval evaluation file must be a JSON object: {path}")
    return payload


def _parse_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []

    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            parsed.append(item)
            continue
        if isinstance(item, str) and item.strip().isdigit():
            parsed.append(int(item.strip()))
    return parsed


def _parse_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def load_retrieval_benchmark(path: Path) -> list[RetrievalBenchmarkItem]:
    payload = _load_json_object(path)
    items_obj = payload.get("items")
    if not isinstance(items_obj, list):
        raise ValueError("Retrieval benchmark must contain an 'items' list")

    items: list[RetrievalBenchmarkItem] = []
    for idx, raw in enumerate(items_obj):
        if not isinstance(raw, dict):
            raise ValueError(f"Benchmark item at index {idx} is not an object")

        meeting_id_obj = raw.get("meeting_id")
        question_obj = raw.get("question")
        if not isinstance(meeting_id_obj, str) or not meeting_id_obj.strip():
            raise ValueError(f"Benchmark item at index {idx} has invalid meeting_id")
        if not isinstance(question_obj, str) or not question_obj.strip():
            raise ValueError(f"Benchmark item at index {idx} has invalid question")

        items.append(
            RetrievalBenchmarkItem(
                meeting_id=meeting_id_obj.strip(),
                question=" ".join(question_obj.split()),
                expected_chunk_ids=_parse_int_list(raw.get("expected_chunk_ids")),
                expected_speaker_labels=_parse_string_list(raw.get("expected_speaker_labels")),
                expected_hints=_parse_string_list(raw.get("expected_hints")),
            )
        )

    return items


def load_retrieval_predictions(path: Path) -> list[RetrievalPredictionItem]:
    payload = _load_json_object(path)
    items_obj = payload.get("items")
    if not isinstance(items_obj, list):
        raise ValueError("Retrieval predictions must contain an 'items' list")

    items: list[RetrievalPredictionItem] = []
    for idx, raw in enumerate(items_obj):
        if not isinstance(raw, dict):
            raise ValueError(f"Prediction item at index {idx} is not an object")

        meeting_id_obj = raw.get("meeting_id")
        question_obj = raw.get("question")
        if not isinstance(meeting_id_obj, str) or not meeting_id_obj.strip():
            raise ValueError(f"Prediction item at index {idx} has invalid meeting_id")
        if not isinstance(question_obj, str) or not question_obj.strip():
            raise ValueError(f"Prediction item at index {idx} has invalid question")

        rewritten_query_obj = raw.get("rewritten_query")
        rewritten_query = (
            " ".join(rewritten_query_obj.split())
            if isinstance(rewritten_query_obj, str) and rewritten_query_obj.strip()
            else ""
        )

        retrieved_obj = raw.get("retrieved")
        if not isinstance(retrieved_obj, list):
            retrieved_obj = []

        retrieved: list[RetrievedEvidence] = []
        for evidence_raw in retrieved_obj:
            if not isinstance(evidence_raw, dict):
                continue

            chunk_id_obj = evidence_raw.get("chunk_id")
            chunk_id: int | None
            if isinstance(chunk_id_obj, bool):
                chunk_id = None
            elif isinstance(chunk_id_obj, int):
                chunk_id = chunk_id_obj
            elif isinstance(chunk_id_obj, str) and chunk_id_obj.strip().isdigit():
                chunk_id = int(chunk_id_obj.strip())
            else:
                chunk_id = None

            speaker_label_obj = evidence_raw.get("speaker_label")
            content_obj = evidence_raw.get("content")
            similarity_obj = evidence_raw.get("similarity")

            similarity: float | None
            if isinstance(similarity_obj, bool):
                similarity = None
            elif isinstance(similarity_obj, (int, float)):
                similarity = float(similarity_obj)
            else:
                similarity = None

            retrieved.append(
                RetrievedEvidence(
                    chunk_id=chunk_id,
                    speaker_label=(
                        speaker_label_obj.strip()
                        if isinstance(speaker_label_obj, str)
                        else "unknown"
                    ),
                    content=(content_obj if isinstance(content_obj, str) else ""),
                    similarity=similarity,
                )
            )

        items.append(
            RetrievalPredictionItem(
                meeting_id=meeting_id_obj.strip(),
                question=" ".join(question_obj.split()),
                rewritten_query=rewritten_query,
                retrieved=retrieved,
            )
        )

    return items


def evaluate_retrieval_benchmark(
    benchmark_items: list[RetrievalBenchmarkItem],
    prediction_items: list[RetrievalPredictionItem],
    *,
    top_k: int = 5,
) -> RetrievalEvaluationResult:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    prediction_map = {(item.meeting_id, item.question): item for item in prediction_items}
    item_results: list[RetrievalItemResult] = []

    chunk_recalls: list[float] = []
    evidence_hits = 0
    empty_retrievals = 0
    chunk_hit_count = 0
    speaker_hit_count = 0
    hint_hit_count = 0
    missing_predictions = 0
    misses: list[dict[str, str]] = []

    for item in benchmark_items:
        prediction = prediction_map.get((item.meeting_id, item.question))
        if prediction is None:
            missing_predictions += 1
            top_retrieved: list[RetrievedEvidence] = []
        else:
            top_retrieved = prediction.retrieved[:top_k]

        if not top_retrieved:
            empty_retrievals += 1

        expected_chunk_ids = set(item.expected_chunk_ids)
        expected_speakers = {label.lower() for label in item.expected_speaker_labels}
        expected_hints = [hint.lower() for hint in item.expected_hints]

        retrieved_chunk_ids = {
            entry.chunk_id for entry in top_retrieved if entry.chunk_id is not None
        }
        retrieved_speakers = {entry.speaker_label.lower() for entry in top_retrieved}
        retrieved_text = " ".join(entry.content.lower() for entry in top_retrieved)

        chunk_matches = expected_chunk_ids & retrieved_chunk_ids
        chunk_hit = bool(chunk_matches)
        if expected_chunk_ids:
            chunk_recall = safe_divide(len(chunk_matches), len(expected_chunk_ids))
            chunk_recalls.append(chunk_recall)
        else:
            chunk_recall = 0.0

        speaker_hit = bool(expected_speakers & retrieved_speakers) if expected_speakers else False
        hint_hit = (
            any(hint in retrieved_text for hint in expected_hints) if expected_hints else False
        )

        has_expectations = bool(expected_chunk_ids or expected_speakers or expected_hints)
        evidence_hit = (
            (chunk_hit or speaker_hit or hint_hit) if has_expectations else bool(top_retrieved)
        )

        if chunk_hit:
            chunk_hit_count += 1
        if speaker_hit:
            speaker_hit_count += 1
        if hint_hit:
            hint_hit_count += 1
        if evidence_hit:
            evidence_hits += 1
        else:
            misses.append({"meeting_id": item.meeting_id, "question": item.question})

        item_results.append(
            RetrievalItemResult(
                meeting_id=item.meeting_id,
                question=item.question,
                retrieved_count=len(top_retrieved),
                chunk_recall_at_k=chunk_recall,
                chunk_hit=chunk_hit,
                speaker_hit=speaker_hit,
                hint_hit=hint_hit,
                evidence_hit=evidence_hit,
            )
        )

    query_count = len(benchmark_items)
    return RetrievalEvaluationResult(
        query_count=query_count,
        top_k=top_k,
        recall_at_k=safe_divide(sum(chunk_recalls), len(chunk_recalls)),
        evidence_hit_rate=rate(evidence_hits, query_count),
        empty_retrieval_rate=rate(empty_retrievals, query_count),
        chunk_hit_rate=rate(chunk_hit_count, query_count),
        speaker_hit_rate=rate(speaker_hit_count, query_count),
        hint_hit_rate=rate(hint_hit_count, query_count),
        missing_predictions=missing_predictions,
        misses=misses[:10],
        item_results=item_results,
    )
