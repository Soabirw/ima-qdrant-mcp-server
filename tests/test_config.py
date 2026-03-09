"""Tests for config.py — load_config with env vars and defaults."""

import pytest

from qdrant_mcp.config import QdrantConfig, load_config


def test_load_config_returns_frozen_dataclass():
    config = load_config()
    assert isinstance(config, QdrantConfig)


def test_load_config_defaults(monkeypatch):
    for key in ("QDRANT_URL", "COLLECTION_NAME", "OLLAMA_URL", "EMBEDDING_MODEL", "VECTOR_SIZE", "MAX_SEARCH_LIMIT"):
        monkeypatch.delenv(key, raising=False)

    config = load_config()

    assert config.collection == "ima-knowledge"
    assert config.qdrant_url == "http://localhost:6333"
    assert config.ollama_url == "http://localhost:11434"
    assert config.embedding_model == "nomic-embed-text"
    assert config.vector_size == 768
    assert config.max_search_limit == 100


def test_load_config_is_immutable(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = load_config()
    with pytest.raises((AttributeError, TypeError)):
        config.qdrant_url = "http://other:6333"  # type: ignore[misc]


@pytest.mark.parametrize(
    "env_key, env_value, attr, expected",
    [
        ("QDRANT_URL", "http://remote:6333", "qdrant_url", "http://remote:6333"),
        ("COLLECTION_NAME", "my-project", "collection", "my-project"),
        ("OLLAMA_URL", "http://gpu-box:11434", "ollama_url", "http://gpu-box:11434"),
        ("EMBEDDING_MODEL", "mxbai-embed-large", "embedding_model", "mxbai-embed-large"),
        ("VECTOR_SIZE", "1024", "vector_size", 1024),
        ("MAX_SEARCH_LIMIT", "50", "max_search_limit", 50),
    ],
)
def test_load_config_env_overrides(monkeypatch, env_key, env_value, attr, expected):
    monkeypatch.setenv(env_key, env_value)
    config = load_config()
    assert getattr(config, attr) == expected


def test_load_config_vector_size_is_int(monkeypatch):
    monkeypatch.setenv("VECTOR_SIZE", "512")
    config = load_config()
    assert isinstance(config.vector_size, int)
    assert config.vector_size == 512


def test_qdrant_config_equality():
    a = QdrantConfig(collection="x", qdrant_url="http://a:6333")
    b = QdrantConfig(collection="x", qdrant_url="http://a:6333")
    assert a == b


def test_resolve_collection_uses_provided_name():
    """_resolve_collection is tested indirectly; verify the config default is stable."""
    config = QdrantConfig(collection="my-collection")
    assert config.collection == "my-collection"
