"""Tests for server.py — MCP tool functions with mocked dependencies."""

from unittest.mock import AsyncMock, patch

import pytest

from qdrant_mcp.config import QdrantConfig
from qdrant_mcp.server import _resolve_collection, qdrant_find, qdrant_store


SAMPLE_CONFIG = QdrantConfig(
    collection="test-collection",
    qdrant_url="http://localhost:6333",
    ollama_url="http://localhost:11434",
    embedding_model="nomic-embed-text",
    vector_size=4,
)
SAMPLE_VECTOR = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# _resolve_collection (pure function — no mocking needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection_name, expected",
    [
        ("explicit-name", "explicit-name"),
        (None, "test-collection"),
        ("", "test-collection"),  # empty string is falsy — falls back to config
    ],
)
def test_resolve_collection(collection_name, expected):
    result = _resolve_collection(collection_name, SAMPLE_CONFIG)
    assert result == expected


# ---------------------------------------------------------------------------
# qdrant_store
# ---------------------------------------------------------------------------


async def test_qdrant_store_returns_confirmation_message():
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.ensure_collection", new=AsyncMock()),
        patch("qdrant_mcp.server.store_points", new=AsyncMock()),
    ):
        result = await qdrant_store("hello world")

    assert "Stored in 'test-collection'" in result
    assert "id:" in result


async def test_qdrant_store_uses_explicit_collection():
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.ensure_collection", new=AsyncMock()),
        patch("qdrant_mcp.server.store_points", new=AsyncMock()),
    ):
        result = await qdrant_store("hello world", collection_name="custom-col")

    assert "custom-col" in result


async def test_qdrant_store_passes_metadata():
    store_mock = AsyncMock()
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.ensure_collection", new=AsyncMock()),
        patch("qdrant_mcp.server.store_points", new=store_mock),
    ):
        await qdrant_store("hello world", metadata={"source": "test"})

    _url, _collection, points = store_mock.call_args.args
    assert points[0]["payload"]["metadata"] == {"source": "test"}


async def test_qdrant_store_defaults_to_empty_metadata():
    store_mock = AsyncMock()
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.ensure_collection", new=AsyncMock()),
        patch("qdrant_mcp.server.store_points", new=store_mock),
    ):
        await qdrant_store("hello world")

    _url, _collection, points = store_mock.call_args.args
    assert points[0]["payload"]["metadata"] == {}


async def test_qdrant_store_embeds_the_information_text():
    embed_mock = AsyncMock(return_value=[SAMPLE_VECTOR])
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=embed_mock),
        patch("qdrant_mcp.server.ensure_collection", new=AsyncMock()),
        patch("qdrant_mcp.server.store_points", new=AsyncMock()),
    ):
        await qdrant_store("the text to embed")

    embed_mock.assert_called_once_with(
        ["the text to embed"],
        model=SAMPLE_CONFIG.embedding_model,
        ollama_url=SAMPLE_CONFIG.ollama_url,
    )


async def test_qdrant_store_ensures_collection_before_storing():
    ensure_mock = AsyncMock()
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.ensure_collection", new=ensure_mock),
        patch("qdrant_mcp.server.store_points", new=AsyncMock()),
    ):
        await qdrant_store("hello")

    ensure_mock.assert_called_once_with(
        SAMPLE_CONFIG.qdrant_url, "test-collection", SAMPLE_CONFIG.vector_size
    )


# ---------------------------------------------------------------------------
# qdrant_find
# ---------------------------------------------------------------------------


async def test_qdrant_find_returns_no_results_message_when_empty():
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=AsyncMock(return_value=[])),
    ):
        result = await qdrant_find("my query")

    assert "No results found" in result
    assert "test-collection" in result


async def test_qdrant_find_formats_results():
    results = [
        {"content": "hello world", "metadata": {}, "score": 0.95},
        {"content": "another doc", "metadata": {"source": "wiki"}, "score": 0.82},
    ]
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=AsyncMock(return_value=results)),
    ):
        result = await qdrant_find("my query")

    assert "## Result 1 (score: 0.950)" in result
    assert "## Result 2 (score: 0.820)" in result
    assert "hello world" in result
    assert "another doc" in result


async def test_qdrant_find_includes_metadata_in_output():
    results = [{"content": "doc text", "metadata": {"source": "blog", "author": "alice"}, "score": 0.9}]
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=AsyncMock(return_value=results)),
    ):
        result = await qdrant_find("query")

    assert "source: blog" in result
    assert "author: alice" in result


async def test_qdrant_find_omits_metadata_section_when_empty():
    results = [{"content": "doc text", "metadata": {}, "score": 0.9}]
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=AsyncMock(return_value=results)),
    ):
        result = await qdrant_find("query")

    assert "_Metadata:" not in result


async def test_qdrant_find_separates_results_with_dividers():
    results = [
        {"content": "first", "metadata": {}, "score": 0.9},
        {"content": "second", "metadata": {}, "score": 0.8},
    ]
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=AsyncMock(return_value=results)),
    ):
        result = await qdrant_find("query")

    assert "---" in result


async def test_qdrant_find_passes_limit_to_search():
    search_mock = AsyncMock(return_value=[])
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=search_mock),
    ):
        await qdrant_find("query", limit=5)

    _url, _collection, _vector, limit = (
        search_mock.call_args.args + (None,) * 4
    )[:4]
    # limit is passed as keyword arg
    assert search_mock.call_args.kwargs.get("limit") == 5 or limit == 5


async def test_qdrant_find_uses_explicit_collection():
    search_mock = AsyncMock(return_value=[])
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=search_mock),
    ):
        await qdrant_find("query", collection_name="other-col")

    assert "No results found in 'other-col'" or search_mock.call_args.args[1] == "other-col"
    # Verify through the search call args
    assert search_mock.call_args.args[1] == "other-col"


async def test_qdrant_find_clamps_limit_to_max_search_limit():
    config_with_low_max = QdrantConfig(
        collection="test-collection",
        qdrant_url="http://localhost:6333",
        ollama_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        vector_size=4,
        max_search_limit=20,
    )
    search_mock = AsyncMock(return_value=[])
    with (
        patch("qdrant_mcp.server.load_config", return_value=config_with_low_max),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=search_mock),
    ):
        await qdrant_find("query", limit=500)

    assert search_mock.call_args.kwargs.get("limit") == 20


async def test_qdrant_find_does_not_clamp_limit_below_max():
    search_mock = AsyncMock(return_value=[])
    with (
        patch("qdrant_mcp.server.load_config", return_value=SAMPLE_CONFIG),
        patch("qdrant_mcp.server.embed_texts", new=AsyncMock(return_value=[SAMPLE_VECTOR])),
        patch("qdrant_mcp.server.search_points", new=search_mock),
    ):
        await qdrant_find("query", limit=5)

    assert search_mock.call_args.kwargs.get("limit") == 5
