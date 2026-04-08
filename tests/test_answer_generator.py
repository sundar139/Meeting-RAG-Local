from __future__ import annotations

import pytest

from meeting_pipeline.embeddings.ollama_client import OllamaClientError
from meeting_pipeline.rag.answer_generator import AnswerGenerator
from meeting_pipeline.rag.models import RetrievedChunk


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


def _sample_evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=7,
            meeting_id="m1",
            speaker_label="SPEAKER_01",
            start_time=10.5,
            end_time=16.2,
            content="We decided to launch Friday and Priya will own QA.",
            similarity=0.93,
        )
    ]


def test_answer_generator_returns_structured_sections() -> None:
    response = (
        '{"Summary":"Launch confirmed [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Key Points":"Friday launch [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Decisions":"Ship Friday [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Action Items":"Priya owns QA [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Uncertainties / Missing Evidence":"None noted."}'
    )
    client = FakeChatClient("".join(response))
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="What did we decide?",
        meeting_id="m1",
        rewritten_query="meeting decisions",
        retrieved_evidence=_sample_evidence(),
    )

    assert result.insufficient_context is False
    assert result.sections["Decisions"].startswith("Ship Friday")
    assert client.call_count == 1
    assert client.last_model == "llama-test"
    assert client.last_messages is not None
    assert "Retrieved evidence" in client.last_messages[1]["content"]


def test_answer_generator_handles_empty_evidence_without_model_call() -> None:
    client = FakeChatClient('{"Summary":"unused"}')
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="Any action items?",
        meeting_id="m1",
        rewritten_query="action items",
        retrieved_evidence=[],
    )

    assert result.insufficient_context is True
    assert "No supporting evidence" in result.sections["Key Points"]
    assert client.call_count == 0


def test_answer_generator_marks_insufficient_when_model_says_so() -> None:
    client = FakeChatClient(
        '{"Summary":"Insufficient context to answer.",'
        '"Key Points":"Insufficient evidence for this section.",'
        '"Decisions":"Insufficient evidence for this section.",'
        '"Action Items":"Insufficient evidence for this section.",'
        '"Uncertainties / Missing Evidence":"Insufficient evidence in retrieved chunks."}'
    )
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="What are blockers?",
        meeting_id="m1",
        rewritten_query="blockers",
        retrieved_evidence=_sample_evidence(),
    )

    assert result.insufficient_context is True


def test_answer_generator_wraps_chat_errors() -> None:
    generator = AnswerGenerator(client=RaisingChatClient(), model_name="llama-test")

    with pytest.raises(RuntimeError, match="Answer generation failed"):
        generator.generate(
            user_question="What did we decide?",
            meeting_id="m1",
            rewritten_query="decisions",
            retrieved_evidence=_sample_evidence(),
        )
