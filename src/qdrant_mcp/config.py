import os
from dataclasses import dataclass


@dataclass(frozen=True)
class QdrantConfig:
    collection: str = "ima-knowledge"
    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    vector_size: int = 768
    max_search_limit: int = 100


def load_config() -> QdrantConfig:
    """Load config from env vars with sensible defaults.

    Env vars: QDRANT_URL, COLLECTION_NAME, OLLAMA_URL, EMBEDDING_MODEL, VECTOR_SIZE,
              MAX_SEARCH_LIMIT
    Collection override per-project is handled by Claude passing collection_name
    to each tool call (informed by .qdrant files in the project root).
    """
    defaults = QdrantConfig()

    return QdrantConfig(
        collection=os.environ.get("COLLECTION_NAME", defaults.collection),
        qdrant_url=os.environ.get("QDRANT_URL", defaults.qdrant_url),
        ollama_url=os.environ.get("OLLAMA_URL", defaults.ollama_url),
        embedding_model=os.environ.get("EMBEDDING_MODEL", defaults.embedding_model),
        vector_size=int(os.environ.get("VECTOR_SIZE", str(defaults.vector_size))),
        max_search_limit=int(os.environ.get("MAX_SEARCH_LIMIT", str(defaults.max_search_limit))),
    )
