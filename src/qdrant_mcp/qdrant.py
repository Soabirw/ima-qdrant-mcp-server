import re

import httpx

_VALID_COLLECTION_NAME = re.compile(r'^[a-zA-Z0-9_-]+$')


def sanitize_collection_name(name: str) -> str:
    """Validate a collection name for safe URL path interpolation.

    Raises ValueError if the name contains path separators or characters
    outside the allowed set (alphanumeric, hyphens, underscores).
    """
    if not name or not _VALID_COLLECTION_NAME.match(name):
        raise ValueError(
            f"Invalid collection name {name!r}: only alphanumeric characters, "
            "hyphens, and underscores are allowed."
        )
    return name


async def ensure_collection(
    qdrant_url: str, collection: str, vector_size: int = 768
) -> None:
    """Create collection if it doesn't exist. Unnamed vectors, Cosine distance."""
    sanitize_collection_name(collection)
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        check = await client.get(url)
        if check.status_code == 200:
            return
        payload = {"vectors": {"size": vector_size, "distance": "Cosine"}}
        response = await client.put(url, json=payload)
        response.raise_for_status()


async def store_points(
    qdrant_url: str, collection: str, points: list[dict]
) -> None:
    """Upsert points. Each point: {"id": str, "vector": list[float], "payload": dict}"""
    sanitize_collection_name(collection)
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.put(url, json={"points": points})
        response.raise_for_status()


async def search_points(
    qdrant_url: str, collection: str, vector: list[float], limit: int = 10
) -> list[dict]:
    """Search by vector similarity. Returns list of {"content": str, "metadata": dict, "score": float}"""
    sanitize_collection_name(collection)
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/search"
    async with httpx.AsyncClient(timeout=30.0) as client:
        check = await client.get(f"{qdrant_url.rstrip('/')}/collections/{collection}")
        if check.status_code == 404:
            return []
        response = await client.post(
            url,
            json={"vector": vector, "limit": limit, "with_payload": True},
        )
        response.raise_for_status()
        results = response.json().get("result", [])
        return [
            {
                "content": r["payload"].get("document", ""),
                "metadata": r["payload"].get("metadata", {}),
                "score": r["score"],
            }
            for r in results
        ]


async def scroll_all(qdrant_url: str, collection: str) -> list[dict]:
    """Scroll all points in collection. For audit purposes."""
    sanitize_collection_name(collection)
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/scroll"
    points: list[dict] = []
    offset = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            body: dict = {"limit": 100, "with_payload": True, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            response = await client.post(url, json=body)
            response.raise_for_status()
            data = response.json().get("result", {})
            batch = data.get("points", [])
            points.extend(batch)
            next_offset = data.get("next_page_offset")
            if not next_offset:
                break
            offset = next_offset

    return points


async def delete_collection(qdrant_url: str, collection: str) -> None:
    """Delete a collection. 404 is not an error."""
    sanitize_collection_name(collection)
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.delete(url)
        if response.status_code not in (200, 404):
            response.raise_for_status()
