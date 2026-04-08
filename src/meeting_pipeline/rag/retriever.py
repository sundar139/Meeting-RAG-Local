from __future__ import annotations

import logging
from typing import Protocol

from meeting_pipeline.db.pgvector_search import SimilarChunkResult
from meeting_pipeline.embeddings.embedder import Embedder
from meeting_pipeline.rag.models import QueryRewriteResult, RetrievalBundle, RetrievedChunk
from meeting_pipeline.rag.query_rewriter import QueryRewriter

LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        self._searcher = searcher
        self._query_rewriter = query_rewriter or QueryRewriter()
        self._embedder = embedder or Embedder()

    def retrieve(
        self,
        meeting_id: str,
        user_query: str,
        *,
        conversation_context: list[str] | None = None,
        top_k: int = 5,
    ) -> RetrievalBundle:
        normalized_meeting_id = meeting_id.strip()
        normalized_query = " ".join(user_query.split())

        if not normalized_meeting_id:
            raise ValueError("meeting_id must be a non-empty string")
        if not normalized_query:
            raise ValueError("user_query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        rewrite_result = self._query_rewriter.rewrite(
            latest_user_question=normalized_query,
            conversation_context=conversation_context,
        )

        LOGGER.info(
            "retrieval_start meeting_id=%s top_k=%d fallback_rewrite=%s",
            normalized_meeting_id,
            top_k,
            rewrite_result.used_fallback,
        )

        query_embedding = self._embedder.embed_query(rewrite_result.rewritten_query)
        search_results = self._searcher.search_similar_chunks(
            meeting_id=normalized_meeting_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )

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

        return RetrievalBundle(
            meeting_id=normalized_meeting_id,
            user_query=normalized_query,
            rewritten_query=rewrite_result.rewritten_query,
            top_k_used=top_k,
            results=chunks,
        )
