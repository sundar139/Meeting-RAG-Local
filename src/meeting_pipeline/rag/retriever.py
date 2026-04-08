from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.db.pgvector_search import SimilarChunkResult
from meeting_pipeline.embeddings.embedder import Embedder
from meeting_pipeline.rag.models import (
    ConversationState,
    QueryRewriteResult,
    RetrievalBundle,
    RetrievalMode,
    RetrievedChunk,
)
from meeting_pipeline.rag.query_rewriter import QueryRewriter
from meeting_pipeline.timing import elapsed_ms, now

LOGGER = logging.getLogger(__name__)
_SPEAKER_TOKEN_PATTERN = re.compile(r"\bSPEAKER_\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievalPolicy:
    default_factoid_top_k: int
    speaker_specific_top_k: int
    action_items_or_decisions_top_k: int
    broad_summary_top_k: int
    meta_or_confidence_top_k: int
    broad_summary_max_candidates: int

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalPolicy:
        return cls(
            default_factoid_top_k=settings.default_factoid_top_k,
            speaker_specific_top_k=settings.speaker_specific_top_k,
            action_items_or_decisions_top_k=settings.action_items_or_decisions_top_k,
            broad_summary_top_k=settings.broad_summary_top_k,
            meta_or_confidence_top_k=settings.meta_or_confidence_top_k,
            broad_summary_max_candidates=max(
                settings.broad_summary_max_candidates,
                settings.broad_summary_top_k,
            ),
        )


class QueryRewriterProtocol(Protocol):
    def rewrite(
        self,
        latest_user_question: str,
        conversation_context: list[str] | None = None,
    ) -> QueryRewriteResult: ...


class QueryEmbedderProtocol(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class VectorSearcherProtocol(Protocol):
    def search_similar_chunks(
        self, meeting_id: str, query_embedding: list[float], top_k: int = 10
    ) -> list[SimilarChunkResult]: ...


class Retriever:
    def __init__(
        self,
        searcher: VectorSearcherProtocol,
        *,
        query_rewriter: QueryRewriterProtocol | None = None,
        embedder: QueryEmbedderProtocol | None = None,
        policy: RetrievalPolicy | None = None,
        settings: Settings | None = None,
    ) -> None:
        runtime_settings = settings or get_settings()
        self._searcher = searcher
        self._query_rewriter = query_rewriter or QueryRewriter()
        self._embedder = embedder or Embedder()
        self._policy = policy or RetrievalPolicy.from_settings(runtime_settings)

    def _classify_retrieval_mode(
        self,
        user_query: str,
        rewrite_result: QueryRewriteResult,
    ) -> RetrievalMode:
        lower_query = user_query.lower()

        if rewrite_result.question_relation == "meta_chat_scope":
            return "meta_or_confidence"

        if any(
            phrase in lower_query
            for phrase in (
                "confidence",
                "uncertain",
                "missing evidence",
                "what can you not",
                "which part is unsupported",
            )
        ):
            return "meta_or_confidence"

        if _SPEAKER_TOKEN_PATTERN.search(user_query) or "specific speaker" in lower_query:
            return "speaker_specific"

        if any(
            phrase in lower_query
            for phrase in (
                "action item",
                "next step",
                "follow-up",
                "follow up",
                "owner",
                "deadline",
                "decision",
                "decided",
            )
        ):
            return "action_items_or_decisions"

        if any(
            phrase in lower_query
            for phrase in (
                "summarize the whole meeting",
                "summary of the meeting",
                "high-level summary",
                "overall summary",
                "what did they discuss",
                "main themes",
                "broad summary",
            )
        ):
            return "broad_summary"

        return "default_factoid"

    def _extract_speaker_filter(self, text: str) -> str | None:
        match = _SPEAKER_TOKEN_PATTERN.search(text)
        return match.group(0).upper() if match else None

    def _top_k_for_mode(self, mode: RetrievalMode) -> int:
        if mode == "speaker_specific":
            return self._policy.speaker_specific_top_k
        if mode == "action_items_or_decisions":
            return self._policy.action_items_or_decisions_top_k
        if mode == "broad_summary":
            return self._policy.broad_summary_top_k
        if mode == "meta_or_confidence":
            return self._policy.meta_or_confidence_top_k
        return self._policy.default_factoid_top_k

    def _diversify_for_broad_summary(
        self,
        results: list[SimilarChunkResult],
        final_top_k: int,
    ) -> list[SimilarChunkResult]:
        if len(results) <= final_top_k:
            return results

        selected: list[SimilarChunkResult] = []
        selected_ids: set[int] = set()
        seen_speakers: set[str] = set()

        # First pass: maximize speaker diversity.
        for result in results:
            speaker = (result.speaker_label or "").strip().upper()
            if speaker and speaker not in seen_speakers:
                selected.append(result)
                selected_ids.add(result.chunk_id)
                seen_speakers.add(speaker)
            if len(selected) >= final_top_k:
                return selected

        # Second pass: maximize temporal spread in 2-minute windows.
        seen_windows: set[int] = set(int(item.start_time // 120) for item in selected)
        for result in results:
            if result.chunk_id in selected_ids:
                continue
            time_window = int(result.start_time // 120)
            if time_window in seen_windows:
                continue
            selected.append(result)
            selected_ids.add(result.chunk_id)
            seen_windows.add(time_window)
            if len(selected) >= final_top_k:
                return selected

        for result in results:
            if result.chunk_id in selected_ids:
                continue
            selected.append(result)
            if len(selected) >= final_top_k:
                break

        return selected

    def retrieve(
        self,
        meeting_id: str,
        user_query: str,
        *,
        conversation_context: list[str] | None = None,
        top_k: int | None = None,
        conversation_state: ConversationState | None = None,
    ) -> RetrievalBundle:
        request_started_at = now()
        normalized_meeting_id = meeting_id.strip()
        normalized_query = " ".join(user_query.split())

        if not normalized_meeting_id:
            raise ValueError("meeting_id must be a non-empty string")
        if not normalized_query:
            raise ValueError("user_query must be a non-empty string")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        rewrite_started_at = now()
        rewrite_result = self._query_rewriter.rewrite(
            latest_user_question=normalized_query,
            conversation_context=conversation_context,
        )
        rewrite_elapsed_ms = elapsed_ms(rewrite_started_at)

        retrieval_mode = self._classify_retrieval_mode(normalized_query, rewrite_result)
        speaker_filter = self._extract_speaker_filter(rewrite_result.rewritten_query)
        policy_top_k = self._top_k_for_mode(retrieval_mode)
        final_top_k = top_k or policy_top_k

        if (
            retrieval_mode == "meta_or_confidence"
            and conversation_state is not None
            and conversation_state.latest_bundle is not None
            and conversation_state.latest_bundle.meeting_id == normalized_meeting_id
            and conversation_state.latest_bundle.results
        ):
            cached_chunks = conversation_state.latest_bundle.results[:final_top_k]
            LOGGER.info(
                "retrieval_cached_context meeting_id=%s mode=%s top_k=%d",
                normalized_meeting_id,
                retrieval_mode,
                final_top_k,
            )
            timings_ms = {
                "query_rewrite": rewrite_elapsed_ms,
                "embedding_query_prep": 0.0,
                "retrieval": 0.0,
                "retrieval_total": elapsed_ms(request_started_at),
            }
            return RetrievalBundle(
                meeting_id=normalized_meeting_id,
                user_query=normalized_query,
                rewritten_query=rewrite_result.rewritten_query,
                top_k_used=min(final_top_k, len(cached_chunks)),
                results=cached_chunks,
                retrieval_mode=retrieval_mode,
                question_relation=rewrite_result.question_relation,
                used_cached_context=True,
                speaker_filter=speaker_filter,
                service_metadata={"timings_ms": timings_ms},
            )

        LOGGER.info(
            "retrieval_start meeting_id=%s mode=%s top_k=%d fallback_rewrite=%s",
            normalized_meeting_id,
            retrieval_mode,
            final_top_k,
            rewrite_result.used_fallback,
        )

        embed_started_at = now()
        query_embedding = self._embedder.embed_query(rewrite_result.rewritten_query)
        embed_elapsed_ms = elapsed_ms(embed_started_at)

        search_top_k = final_top_k
        if retrieval_mode == "broad_summary":
            search_top_k = max(
                final_top_k,
                self._policy.broad_summary_max_candidates,
            )

        retrieval_started_at = now()
        search_results = self._searcher.search_similar_chunks(
            meeting_id=normalized_meeting_id,
            query_embedding=query_embedding,
            top_k=search_top_k,
        )

        if retrieval_mode == "speaker_specific" and speaker_filter:
            search_results = [
                result
                for result in search_results
                if (result.speaker_label or "").strip().upper() == speaker_filter
            ]

        if retrieval_mode == "broad_summary":
            search_results = self._diversify_for_broad_summary(search_results, final_top_k)
        elif len(search_results) > final_top_k:
            search_results = search_results[:final_top_k]

        retrieval_elapsed_ms = elapsed_ms(retrieval_started_at)

        chunks = [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                meeting_id=result.meeting_id,
                speaker_label=result.speaker_label,
                start_time=result.start_time,
                end_time=result.end_time,
                content=result.content,
                similarity=result.similarity,
            )
            for result in search_results
        ]

        timings_ms = {
            "query_rewrite": rewrite_elapsed_ms,
            "embedding_query_prep": embed_elapsed_ms,
            "retrieval": retrieval_elapsed_ms,
            "retrieval_total": elapsed_ms(request_started_at),
        }

        return RetrievalBundle(
            meeting_id=normalized_meeting_id,
            user_query=normalized_query,
            rewritten_query=rewrite_result.rewritten_query,
            top_k_used=final_top_k,
            results=chunks,
            retrieval_mode=retrieval_mode,
            question_relation=rewrite_result.question_relation,
            used_cached_context=False,
            speaker_filter=speaker_filter,
            service_metadata={"timings_ms": timings_ms},
        )
