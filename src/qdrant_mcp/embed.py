import httpx


async def embed_texts(
    texts: list[str],
    model: str = "nomic-embed-text",
    ollama_url: str = "http://localhost:11434",
) -> list[list[float]]:
    """Embed texts via Ollama HTTP API. Pure function: texts in, vectors out."""
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
