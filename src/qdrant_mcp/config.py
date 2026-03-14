import os
from dataclasses import dataclass

_PROVIDER_DEFAULTS: dict[str, dict[str, str | int]] = {
    "ollama": {"embedding_model": "nomic-embed-text", "vector_size": 768},
    "fastembed": {"embedding_model": "BAAI/bge-small-en-v1.5", "vector_size": 384},
}


@dataclass(frozen=True)
class QdrantConfig:
    collection: str = "ima-knowledge"
    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    vector_size: int = 768
    max_search_limit: int = 100


def load_config() -> QdrantConfig:
    """Load config from env vars with sensible defaults.

    Env vars: QDRANT_URL, COLLECTION_NAME, OLLAMA_URL, EMBEDDING_PROVIDER,
              EMBEDDING_MODEL, VECTOR_SIZE, MAX_SEARCH_LIMIT

    When EMBEDDING_PROVIDER is set, model and vector_size auto-default to
    provider-appropriate values unless explicitly overridden.
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "ollama")
    pdefaults = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["ollama"])

    model = os.environ.get("EMBEDDING_MODEL") or str(pdefaults["embedding_model"])
    vector_size = int(os.environ.get("VECTOR_SIZE") or pdefaults["vector_size"])

    return QdrantConfig(
        collection=os.environ.get("COLLECTION_NAME", "ima-knowledge"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        embedding_provider=provider,
        embedding_model=model,
        vector_size=vector_size,
        max_search_limit=int(os.environ.get("MAX_SEARCH_LIMIT", "100")),
    )
