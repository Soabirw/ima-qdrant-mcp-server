import asyncio

import httpx

_fastembed_model = None


async def embed_texts(
    texts: list[str],
    model: str = "nomic-embed-text",
    ollama_url: str = "http://localhost:11434",
    provider: str = "ollama",
) -> list[list[float]]:
    """Embed texts via Ollama or fastembed. Pure function: texts in, vectors out."""
    if provider == "fastembed":
        return await _embed_fastembed(texts, model)
    return await _embed_ollama(texts, model, ollama_url)


async def _embed_ollama(
    texts: list[str], model: str, ollama_url: str
) -> list[list[float]]:
    url = f"{ollama_url.rstrip('/')}/api/embed"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json={"model": model, "input": texts})
            response.raise_for_status()
            data = response.json()
            if "embeddings" not in data:
                raise ValueError(f"Ollama response missing 'embeddings' key: {data}")
            return data["embeddings"]
    except httpx.ConnectError as e:
        raise RuntimeError(f"Cannot reach Ollama at {ollama_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{model}' not found. Run: ollama pull {model}"
            ) from e
        raise RuntimeError(f"Ollama request failed: {e}") from e


def _get_fastembed_model(model: str):
    global _fastembed_model
    if _fastembed_model is not None and _fastembed_model._model_name == model:
        return _fastembed_model
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise RuntimeError(
            "fastembed not installed. Run: pip install qdrant-mcp[fastembed]"
        ) from e
    _fastembed_model = TextEmbedding(model_name=model)
    _fastembed_model._model_name = model
    return _fastembed_model


async def _embed_fastembed(texts: list[str], model: str) -> list[list[float]]:
    fe_model = _get_fastembed_model(model)
    return await asyncio.to_thread(
        lambda: [v.tolist() for v in fe_model.embed(texts)]
    )
