from __future__ import annotations

import pytest

from meeting_pipeline.embeddings.ollama_client import OllamaClientError
from meeting_pipeline.rag.query_rewriter import QueryRewriter


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_model: str | None = None
        self.last_messages: list[dict[str, str]] | None = None

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        self.last_model = model
        self.last_messages = messages
        return self.response


class RaisingChatClient:
    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        _ = model
        _ = messages
        raise OllamaClientError("offline")


def test_query_rewriter_returns_model_rewrite() -> None:
    client = FakeChatClient("  final rewritten query  ")
    rewriter = QueryRewriter(client=client, model_name="llama-test")

    result = rewriter.rewrite(
        "What did we commit to?",
        conversation_context=["Earlier they discussed release date"],
    )

    assert result.original_query == "What did we commit to?"
    assert result.rewritten_query == "final rewritten query"
    assert result.used_fallback is False
    assert client.last_model == "llama-test"
    assert client.last_messages is not None
    assert client.last_messages[1]["content"].count("Earlier they discussed release date") == 1


def test_query_rewriter_uses_fallback_on_client_error() -> None:
    rewriter = QueryRewriter(client=RaisingChatClient(), model_name="llama-test")

    result = rewriter.rewrite("What changed yesterday?")

    assert result.original_query == "What changed yesterday?"
    assert result.rewritten_query == "What changed yesterday?"
    assert result.used_fallback is True


def test_query_rewriter_rejects_empty_question() -> None:
    rewriter = QueryRewriter(client=FakeChatClient("ok"), model_name="llama-test")

    with pytest.raises(ValueError):
        rewriter.rewrite("   ")
