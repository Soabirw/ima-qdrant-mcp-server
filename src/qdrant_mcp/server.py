import uuid

from fastmcp import FastMCP

from .config import load_config
from .embed import embed_texts
from .qdrant import ensure_collection, search_points, store_points

mcp = FastMCP("qdrant-mcp", instructions="Qdrant semantic memory with Ollama or fastembed embeddings")


def _resolve_collection(collection_name: str | None, config) -> str:
    return collection_name or config.collection


@mcp.tool()
async def qdrant_store(
    information: str,
    collection_name: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Store information in Qdrant for later semantic retrieval.

    Args:
        information: The text to store
        collection_name: Target collection (default: from .qdrant config or ima-knowledge)
        metadata: Optional metadata dict to attach
    """
    config = load_config()
    collection = _resolve_collection(collection_name, config)

    vectors = await embed_texts([information], model=config.embedding_model, ollama_url=config.ollama_url, provider=config.embedding_provider)
    vector = vectors[0]

    await ensure_collection(config.qdrant_url, collection, config.vector_size)

    point_id = str(uuid.uuid4())
    point = {
        "id": point_id,
        "vector": vector,
        "payload": {"document": information, "metadata": metadata or {}},
    }
    await store_points(config.qdrant_url, collection, [point])

    return f"Stored in '{collection}' (id: {point_id})"


@mcp.tool()
async def qdrant_find(
    query: str,
    collection_name: str | None = None,
    limit: int = 10,
) -> str:
    """Find relevant information in Qdrant by semantic search.

    Args:
        query: Search query
        collection_name: Target collection (default: from .qdrant config or ima-knowledge)
        limit: Max results to return (default: 10)
    """
    config = load_config()
    collection = _resolve_collection(collection_name, config)
    clamped_limit = min(limit, config.max_search_limit)

    vectors = await embed_texts([query], model=config.embedding_model, ollama_url=config.ollama_url, provider=config.embedding_provider)
    vector = vectors[0]

    results = await search_points(config.qdrant_url, collection, vector, limit=clamped_limit)

    if not results:
        return f"No results found in '{collection}'."

    sections = []
    for i, result in enumerate(results, 1):
        score = f"{result['score']:.3f}"
        content = result["content"]
        meta = result["metadata"]
        header = f"## Result {i} (score: {score})"
        body = content
        if meta:
            meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            body = f"{content}\n\n_Metadata: {meta_str}_"
        sections.append(f"{header}\n\n{body}")

    return "\n\n---\n\n".join(sections)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
