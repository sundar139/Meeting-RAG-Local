from __future__ import annotations

import logging
from collections.abc import Sequence

from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.embeddings.ollama_client import OllamaClient, OllamaClientError
from meeting_pipeline.rag.models import QueryRewriteResult

LOGGER = logging.getLogger(__name__)
DEFAULT_CHAT_MODEL = "llama3.2:3b-instruct"


class QueryRewriter:
    def __init__(
        self,
        client: OllamaClient | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        runtime_settings = settings or get_settings()
        self._client = client or OllamaClient.from_settings(runtime_settings)
        configured_model = model_name or runtime_settings.ollama_chat_model
        normalized_model = str(configured_model).strip()
        self._model_name = normalized_model or DEFAULT_CHAT_MODEL

    def rewrite(
        self,
        latest_user_question: str,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewriteResult:
        normalized_question = " ".join(latest_user_question.split())
        if not normalized_question:
            raise ValueError("latest_user_question must be a non-empty string")

        context_lines = [
            " ".join(item.split()) for item in conversation_context or [] if item.strip()
        ]
        context_block = "\n".join(f"- {item}" for item in context_lines)

        system_prompt = (
            "Rewrite the latest user question into a standalone retrieval query for a meeting "
            "transcript search system. Preserve names, dates, speaker references, and important "
            "follow-up context. Do not answer the question. Do not invent facts. Return only the "
            "rewritten query text."
        )

        user_prompt = (
            f"Latest question:\n{normalized_question}\n\n"
            f"Recent conversation context (optional):\n{context_block or '- none'}"
        )

        LOGGER.info("Rewriting query context_messages=%d", len(context_lines))
        try:
            rewritten = self._client.chat(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            normalized_rewrite = " ".join(rewritten.split())
            if not normalized_rewrite:
                raise ValueError("empty rewrite")

            return QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_rewrite,
                used_fallback=False,
            )
        except (OllamaClientError, ValueError) as exc:
            LOGGER.warning("Query rewriting failed; using fallback query. reason=%s", exc)
            return QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_question,
                used_fallback=True,
            )
