"""Shared fixtures for qdrant-mcp-server tests."""

import pytest


QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION = "test-collection"
VECTOR_SIZE = 4  # Small vectors for tests
SAMPLE_VECTOR = [0.1, 0.2, 0.3, 0.4]
SAMPLE_TEXTS = ["hello world", "semantic search"]
SAMPLE_VECTORS = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]


@pytest.fixture
def qdrant_url():
    return QDRANT_URL


@pytest.fixture
def ollama_url():
    return OLLAMA_URL


@pytest.fixture
def collection():
    return COLLECTION


@pytest.fixture
def sample_vector():
    return SAMPLE_VECTOR


@pytest.fixture
def sample_vectors():
    return SAMPLE_VECTORS


@pytest.fixture
def sample_texts():
    return SAMPLE_TEXTS


@pytest.fixture
def sample_point():
    return {
        "id": "abc-123",
        "vector": SAMPLE_VECTOR,
        "payload": {"document": "hello world", "metadata": {}},
    }


@pytest.fixture
def qdrant_search_result():
    """A single Qdrant search result as returned by the API."""
    return {
        "id": "abc-123",
        "score": 0.95,
        "payload": {"document": "hello world", "metadata": {"source": "test"}},
    }


@pytest.fixture
def qdrant_collection_exists_response():
    """200 response body when a collection exists."""
    return {"status": "ok", "result": {"name": COLLECTION}}


@pytest.fixture
def qdrant_collection_missing_response():
    """404 response body when a collection does not exist."""
    return {"status": {"error": "Not found"}}


@pytest.fixture
def ollama_embed_response():
    """Successful Ollama /api/embed response body."""
    return {"embeddings": SAMPLE_VECTORS}
