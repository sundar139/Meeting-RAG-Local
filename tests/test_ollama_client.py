from __future__ import annotations

import pytest

from meeting_pipeline.embeddings.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaMalformedResponseError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)


class FakeClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def embed(self, model: str, input: str | list[str]) -> object:
        _ = model
        _ = input
        return self.payload


class FakeChatClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def chat(self, model: str, messages: list[dict[str, str]]) -> object:
        _ = model
        _ = messages
        return self.payload


class FakeTypedResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, object]:
        return self._payload


def test_ollama_client_embed_parses_valid_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeClient(payload={"embedding": [0.1] * 768}),
    )

    client = OllamaClient(host="http://localhost:11434")
    embedding = client.embed(model="nomic", text="hello")

    assert len(embedding) == 768


def test_ollama_client_embed_parses_typed_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeClient(payload=FakeTypedResponse({"embeddings": [[0.1] * 768]})),
    )

    client = OllamaClient(host="http://localhost:11434")
    embedding = client.embed(model="nomic", text="hello")

    assert len(embedding) == 768


def test_ollama_client_raises_on_malformed_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeClient(payload={"unexpected": []}),
    )

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaMalformedResponseError):
        client.embed(model="nomic", text="hello")


def test_ollama_client_maps_model_not_found_error(monkeypatch) -> None:
    class RaisingClient:
        def embed(self, model: str, input: str | list[str]) -> object:
            _ = model
            _ = input
            raise RuntimeError("model 'missing' not found")

    monkeypatch.setattr(OllamaClient, "_build_client", lambda self: RaisingClient())

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaModelNotFoundError):
        client.embed(model="missing", text="hello")


def test_ollama_client_maps_unavailable_error(monkeypatch) -> None:
    class RaisingClient:
        def embed(self, model: str, input: str | list[str]) -> object:
            _ = model
            _ = input
            raise RuntimeError("connection refused")

    monkeypatch.setattr(OllamaClient, "_build_client", lambda self: RaisingClient())

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaUnavailableError):
        client.embed(model="nomic", text="hello")


def test_ollama_client_wraps_unknown_errors(monkeypatch) -> None:
    class RaisingClient:
        def embed(self, model: str, input: str | list[str]) -> object:
            _ = model
            _ = input
            raise RuntimeError("boom")

    monkeypatch.setattr(OllamaClient, "_build_client", lambda self: RaisingClient())

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaClientError):
        client.embed(model="nomic", text="hello")


def test_ollama_client_chat_parses_message_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeChatClient(payload={"message": {"content": "hello world"}}),
    )

    client = OllamaClient(host="http://localhost:11434")
    response = client.chat(
        model="llama-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response == "hello world"


def test_ollama_client_chat_parses_typed_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeChatClient(
            payload=FakeTypedResponse({"message": {"content": "typed hello"}})
        ),
    )

    client = OllamaClient(host="http://localhost:11434")
    response = client.chat(
        model="llama-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response == "typed hello"


def test_ollama_client_chat_parses_response_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeChatClient(payload={"response": "fallback text"}),
    )

    client = OllamaClient(host="http://localhost:11434")
    response = client.chat(
        model="llama-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response == "fallback text"


def test_ollama_client_chat_raises_on_malformed_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeChatClient(payload={"message": {"unexpected": "value"}}),
    )

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaMalformedResponseError):
        client.chat(
            model="llama-test",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_ollama_client_chat_maps_model_not_found_error(monkeypatch) -> None:
    class RaisingChatClient:
        def chat(self, model: str, messages: list[dict[str, str]]) -> object:
            _ = model
            _ = messages
            raise RuntimeError("model 'missing' not found")

    monkeypatch.setattr(OllamaClient, "_build_client", lambda self: RaisingChatClient())

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaModelNotFoundError):
        client.chat(
            model="missing",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_ollama_client_chat_maps_unavailable_error(monkeypatch) -> None:
    class RaisingChatClient:
        def chat(self, model: str, messages: list[dict[str, str]]) -> object:
            _ = model
            _ = messages
            raise RuntimeError("connection refused")

    monkeypatch.setattr(OllamaClient, "_build_client", lambda self: RaisingChatClient())

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaUnavailableError):
        client.chat(
            model="llama-test",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_ollama_client_chat_wraps_unknown_errors(monkeypatch) -> None:
    class RaisingChatClient:
        def chat(self, model: str, messages: list[dict[str, str]]) -> object:
            _ = model
            _ = messages
            raise RuntimeError("boom")

    monkeypatch.setattr(OllamaClient, "_build_client", lambda self: RaisingChatClient())

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(OllamaClientError):
        client.chat(
            model="llama-test",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_ollama_client_chat_requires_messages(monkeypatch) -> None:
    monkeypatch.setattr(
        OllamaClient,
        "_build_client",
        lambda self: FakeChatClient(payload={"message": {"content": "hello"}}),
    )

    client = OllamaClient(host="http://localhost:11434")
    with pytest.raises(ValueError):
        client.chat(model="llama-test", messages=[])
