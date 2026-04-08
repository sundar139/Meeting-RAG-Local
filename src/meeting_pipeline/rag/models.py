from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    used_fallback: bool


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    meeting_id: str
    speaker_label: str
    start_time: float
    end_time: float
    content: str
    similarity: float


@dataclass(frozen=True)
class RetrievalBundle:
    meeting_id: str
    user_query: str
    rewritten_query: str
    top_k_used: int
    results: list[RetrievedChunk]


@dataclass(frozen=True)
class GroundedAnswerResult:
    meeting_id: str
    question: str
    rewritten_query: str
    sections: dict[str, str]
    raw_answer: str
    insufficient_context: bool
