from __future__ import annotations

import pytest

from meeting_pipeline.config import Settings
from meeting_pipeline.embeddings.ollama_client import OllamaClientError
from meeting_pipeline.rag.query_rewriter import QueryRewriter


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0
        self.last_model: str | None = None
        self.last_messages: list[dict[str, str]] | None = None

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
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
    assert result.question_relation == "standalone_direct"
    assert client.last_model == "llama-test"
    assert client.last_messages is not None
    assert client.last_messages[1]["content"].count("Earlier they discussed release date") == 1


def test_query_rewriter_uses_fallback_on_client_error() -> None:
    rewriter = QueryRewriter(client=RaisingChatClient(), model_name="llama-test")

    result = rewriter.rewrite("What changed yesterday?")

    assert result.original_query == "What changed yesterday?"
    assert result.rewritten_query == "What changed yesterday?"
    assert result.used_fallback is True


def test_query_rewriter_preserves_original_when_rewrite_is_lossy() -> None:
    client = FakeChatClient("Summarize the conversation")
    rewriter = QueryRewriter(client=client, model_name="llama-test")

    result = rewriter.rewrite("Give 3 bullet points about SPEAKER_01 decisions in 2024")

    assert result.rewritten_query == "Give 3 bullet points about SPEAKER_01 decisions in 2024"
    assert result.used_fallback is True
    assert result.was_lossy is True


def test_query_rewriter_detects_meta_question_without_model_call() -> None:
    client = FakeChatClient("unused")
    rewriter = QueryRewriter(client=client, model_name="llama-test")

    result = rewriter.rewrite(
        "Which parts of your previous answer are uncertain?",
        conversation_context=["assistant: Prior answer summary"],
    )

    assert result.question_relation == "meta_chat_scope"
    assert result.used_fallback is True
    assert client.call_count == 0


def test_query_rewriter_rejects_empty_question() -> None:
    rewriter = QueryRewriter(client=FakeChatClient("ok"), model_name="llama-test")

    with pytest.raises(ValueError):
        rewriter.rewrite("   ")


def test_query_rewriter_caches_identical_inputs() -> None:
    client = FakeChatClient("rewritten once")
    settings = Settings(_env_file=None, enable_rag_caching=True, query_rewrite_cache_size=16)
    rewriter = QueryRewriter(client=client, model_name="llama-test", settings=settings)

    first = rewriter.rewrite("What did we decide?", conversation_context=["assistant: context"])
    second = rewriter.rewrite("What did we decide?", conversation_context=["assistant: context"])

    assert first.used_cache is False
    assert second.used_cache is True
    assert client.call_count == 1


def test_query_rewriter_fast_mode_can_skip_model_call() -> None:
    client = FakeChatClient("unused")
    settings = Settings(_env_file=None, fast_mode_skip_query_rewrite=True)
    rewriter = QueryRewriter(client=client, model_name="llama-test", settings=settings)

    result = rewriter.rewrite("What decisions were made?", fast_mode=True)

    assert result.rewritten_query == "What decisions were made?"
    assert client.call_count == 0
