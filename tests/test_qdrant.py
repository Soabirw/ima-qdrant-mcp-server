"""Tests for qdrant.py — all Qdrant HTTP operations with mocked transport."""

import pytest
import httpx
from pytest_httpx import HTTPXMock

from qdrant_mcp.qdrant import (
    delete_collection,
    ensure_collection,
    scroll_all,
    search_points,
    store_points,
)


QDRANT_URL = "http://localhost:6333"
COLLECTION = "test-collection"
VECTOR_SIZE = 4
SAMPLE_VECTOR = [0.1, 0.2, 0.3, 0.4]
COLLECTION_URL = f"{QDRANT_URL}/collections/{COLLECTION}"


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


async def test_ensure_collection_skips_create_when_exists(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=200, json={"status": "ok"})

    await ensure_collection(QDRANT_URL, COLLECTION, VECTOR_SIZE)

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    assert requests[0].method == "GET"


async def test_ensure_collection_creates_when_missing(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=404, json={})
    httpx_mock.add_response(method="PUT", url=COLLECTION_URL, status_code=200, json={"status": "ok"})

    await ensure_collection(QDRANT_URL, COLLECTION, VECTOR_SIZE)

    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    assert requests[0].method == "GET"
    assert requests[1].method == "PUT"


async def test_ensure_collection_sends_correct_vector_config(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=404, json={})
    httpx_mock.add_response(method="PUT", url=COLLECTION_URL, status_code=200, json={"status": "ok"})

    await ensure_collection(QDRANT_URL, COLLECTION, vector_size=512)

    import json
    put_request = httpx_mock.get_requests()[1]
    body = json.loads(put_request.content)
    assert body["vectors"]["size"] == 512
    assert body["vectors"]["distance"] == "Cosine"


async def test_ensure_collection_strips_trailing_slash(httpx_mock: HTTPXMock):
    url = f"{QDRANT_URL}/collections/{COLLECTION}"
    httpx_mock.add_response(method="GET", url=url, status_code=200, json={"status": "ok"})

    await ensure_collection(f"{QDRANT_URL}/", COLLECTION, VECTOR_SIZE)

    assert len(httpx_mock.get_requests()) == 1


# ---------------------------------------------------------------------------
# store_points
# ---------------------------------------------------------------------------


async def test_store_points_upserts_successfully(httpx_mock: HTTPXMock):
    points_url = f"{QDRANT_URL}/collections/{COLLECTION}/points"
    httpx_mock.add_response(method="PUT", url=points_url, status_code=200, json={"status": "ok"})

    points = [{"id": "abc", "vector": SAMPLE_VECTOR, "payload": {"document": "hi", "metadata": {}}}]
    await store_points(QDRANT_URL, COLLECTION, points)

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    assert requests[0].method == "PUT"


async def test_store_points_sends_all_points(httpx_mock: HTTPXMock):
    points_url = f"{QDRANT_URL}/collections/{COLLECTION}/points"
    httpx_mock.add_response(method="PUT", url=points_url, status_code=200, json={"status": "ok"})

    import json
    points = [
        {"id": "a", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"document": "one", "metadata": {}}},
        {"id": "b", "vector": [0.5, 0.6, 0.7, 0.8], "payload": {"document": "two", "metadata": {}}},
    ]
    await store_points(QDRANT_URL, COLLECTION, points)

    body = json.loads(httpx_mock.get_requests()[0].content)
    assert len(body["points"]) == 2
    assert body["points"][0]["id"] == "a"
    assert body["points"][1]["id"] == "b"


async def test_store_points_raises_on_http_error(httpx_mock: HTTPXMock):
    points_url = f"{QDRANT_URL}/collections/{COLLECTION}/points"
    httpx_mock.add_response(method="PUT", url=points_url, status_code=500, json={"error": "server error"})

    with pytest.raises(httpx.HTTPStatusError):
        await store_points(QDRANT_URL, COLLECTION, [{"id": "x", "vector": SAMPLE_VECTOR, "payload": {}}])


# ---------------------------------------------------------------------------
# search_points
# ---------------------------------------------------------------------------


async def test_search_points_returns_empty_list_when_collection_missing(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=404, json={})

    result = await search_points(QDRANT_URL, COLLECTION, SAMPLE_VECTOR, limit=10)

    assert result == []


async def test_search_points_returns_parsed_results(httpx_mock: HTTPXMock):
    search_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=200, json={"status": "ok"})
    httpx_mock.add_response(
        method="POST",
        url=search_url,
        json={
            "result": [
                {
                    "id": "abc",
                    "score": 0.95,
                    "payload": {"document": "hello world", "metadata": {"source": "test"}},
                }
            ]
        },
    )

    result = await search_points(QDRANT_URL, COLLECTION, SAMPLE_VECTOR, limit=5)

    assert len(result) == 1
    assert result[0]["content"] == "hello world"
    assert result[0]["metadata"] == {"source": "test"}
    assert result[0]["score"] == pytest.approx(0.95)


async def test_search_points_handles_empty_results(httpx_mock: HTTPXMock):
    search_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=200, json={"status": "ok"})
    httpx_mock.add_response(method="POST", url=search_url, json={"result": []})

    result = await search_points(QDRANT_URL, COLLECTION, SAMPLE_VECTOR, limit=10)

    assert result == []


async def test_search_points_handles_missing_payload_keys(httpx_mock: HTTPXMock):
    search_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=200, json={"status": "ok"})
    httpx_mock.add_response(
        method="POST",
        url=search_url,
        json={"result": [{"id": "xyz", "score": 0.7, "payload": {}}]},
    )

    result = await search_points(QDRANT_URL, COLLECTION, SAMPLE_VECTOR)

    assert result[0]["content"] == ""
    assert result[0]["metadata"] == {}


async def test_search_points_sends_correct_payload(httpx_mock: HTTPXMock):
    import json as _json
    search_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    httpx_mock.add_response(method="GET", url=COLLECTION_URL, status_code=200, json={"status": "ok"})
    httpx_mock.add_response(method="POST", url=search_url, json={"result": []})

    await search_points(QDRANT_URL, COLLECTION, SAMPLE_VECTOR, limit=7)

    post_request = httpx_mock.get_requests()[1]
    body = _json.loads(post_request.content)
    assert body["vector"] == SAMPLE_VECTOR
    assert body["limit"] == 7
    assert body["with_payload"] is True


# ---------------------------------------------------------------------------
# scroll_all
# ---------------------------------------------------------------------------


async def test_scroll_all_returns_single_page(httpx_mock: HTTPXMock):
    scroll_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll"
    httpx_mock.add_response(
        method="POST",
        url=scroll_url,
        json={"result": {"points": [{"id": "a", "payload": {"document": "doc1"}}], "next_page_offset": None}},
    )

    result = await scroll_all(QDRANT_URL, COLLECTION)

    assert len(result) == 1
    assert result[0]["id"] == "a"


async def test_scroll_all_paginates_until_no_offset(httpx_mock: HTTPXMock):
    scroll_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll"
    httpx_mock.add_response(
        method="POST",
        url=scroll_url,
        json={"result": {"points": [{"id": "a"}], "next_page_offset": "cursor-1"}},
    )
    httpx_mock.add_response(
        method="POST",
        url=scroll_url,
        json={"result": {"points": [{"id": "b"}], "next_page_offset": None}},
    )

    result = await scroll_all(QDRANT_URL, COLLECTION)

    assert len(result) == 2
    assert result[0]["id"] == "a"
    assert result[1]["id"] == "b"


async def test_scroll_all_returns_empty_list_for_empty_collection(httpx_mock: HTTPXMock):
    scroll_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll"
    httpx_mock.add_response(
        method="POST",
        url=scroll_url,
        json={"result": {"points": [], "next_page_offset": None}},
    )

    result = await scroll_all(QDRANT_URL, COLLECTION)

    assert result == []


async def test_scroll_all_first_request_has_no_offset(httpx_mock: HTTPXMock):
    import json as _json
    scroll_url = f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll"
    httpx_mock.add_response(
        method="POST",
        url=scroll_url,
        json={"result": {"points": [], "next_page_offset": None}},
    )

    await scroll_all(QDRANT_URL, COLLECTION)

    body = _json.loads(httpx_mock.get_requests()[0].content)
    assert "offset" not in body


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


async def test_delete_collection_succeeds_on_200(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="DELETE", url=COLLECTION_URL, status_code=200, json={"status": "ok"})

    await delete_collection(QDRANT_URL, COLLECTION)

    assert len(httpx_mock.get_requests()) == 1


async def test_delete_collection_ignores_404(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="DELETE", url=COLLECTION_URL, status_code=404, json={})

    await delete_collection(QDRANT_URL, COLLECTION)  # Should not raise


async def test_delete_collection_raises_on_other_errors(httpx_mock: HTTPXMock):
    httpx_mock.add_response(method="DELETE", url=COLLECTION_URL, status_code=500, json={"error": "crash"})

    with pytest.raises(httpx.HTTPStatusError):
        await delete_collection(QDRANT_URL, COLLECTION)
