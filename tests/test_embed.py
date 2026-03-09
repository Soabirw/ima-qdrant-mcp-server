"""Tests for embed.py — embed_texts with mocked httpx transport."""

import pytest
import httpx
from pytest_httpx import HTTPXMock

from qdrant_mcp.embed import embed_texts


OLLAMA_URL = "http://localhost:11434"
MODEL = "nomic-embed-text"
TEXTS = ["hello world", "semantic search"]
VECTORS = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]


async def test_embed_texts_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
        json={"embeddings": VECTORS},
    )

    result = await embed_texts(TEXTS, model=MODEL, ollama_url=OLLAMA_URL)

    assert result == VECTORS


async def test_embed_texts_single_input(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
        json={"embeddings": [VECTORS[0]]},
    )

    result = await embed_texts(["hello world"], model=MODEL, ollama_url=OLLAMA_URL)

    assert result == [VECTORS[0]]


async def test_embed_texts_strips_trailing_slash(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
        json={"embeddings": VECTORS},
    )

    result = await embed_texts(TEXTS, model=MODEL, ollama_url=f"{OLLAMA_URL}/")

    assert result == VECTORS


async def test_embed_texts_404_raises_model_not_found(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
        status_code=404,
    )

    with pytest.raises(RuntimeError, match="not found"):
        await embed_texts(TEXTS, model=MODEL, ollama_url=OLLAMA_URL)


async def test_embed_texts_non_404_http_error_raises_runtime(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
        status_code=500,
    )

    with pytest.raises(RuntimeError, match="Ollama request failed"):
        await embed_texts(TEXTS, model=MODEL, ollama_url=OLLAMA_URL)


async def test_embed_texts_connect_error_raises_runtime(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(
        httpx.ConnectError("Connection refused"),
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
    )

    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        await embed_texts(TEXTS, model=MODEL, ollama_url=OLLAMA_URL)


async def test_embed_texts_missing_embeddings_key_raises_value_error(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
        json={"error": "unexpected format"},
    )

    with pytest.raises(ValueError, match="missing 'embeddings' key"):
        await embed_texts(TEXTS, model=MODEL, ollama_url=OLLAMA_URL)


@pytest.mark.parametrize(
    "model_name",
    ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
)
async def test_embed_texts_sends_correct_model(httpx_mock: HTTPXMock, model_name: str):
    httpx_mock.add_response(
        method="POST",
        url=f"{OLLAMA_URL}/api/embed",
        json={"embeddings": VECTORS},
    )

    await embed_texts(TEXTS, model=model_name, ollama_url=OLLAMA_URL)

    request = httpx_mock.get_requests()[0]
    import json
    body = json.loads(request.content)
    assert body["model"] == model_name
    assert body["input"] == TEXTS
