"""Integration tests for POST /api/v1/query."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.retrieval.vector_store import SearchResult


@pytest.mark.asyncio
async def test_query_no_documents(async_client, mock_vector_store):
    mock_vector_store.search = AsyncMock(return_value=[])
    response = await async_client.post(
        "/api/v1/query",
        json={"question": "What is in the documents?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "No relevant documents" in data["answer"]
    assert data["sources"] == []


@pytest.mark.asyncio
async def test_query_with_results(async_client, mock_vector_store, mock_llm):
    mock_vector_store.search = AsyncMock(
        return_value=[
            SearchResult(
                chroma_id="doc1_0",
                document_id="doc-uuid-1",
                chunk_index=0,
                text="The capital of France is Paris.",
                score=0.05,
                page_number=1,
            )
        ]
    )
    mock_llm.generate = AsyncMock(return_value="The capital of France is Paris.")

    response = await async_client.post(
        "/api/v1/query",
        json={"question": "What is the capital of France?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The capital of France is Paris."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc-uuid-1"
    assert data["sources"][0]["relevance_score"] == pytest.approx(0.95, abs=0.01)


@pytest.mark.asyncio
async def test_query_with_document_filter(async_client, mock_vector_store, mock_llm):
    mock_vector_store.search = AsyncMock(return_value=[])
    response = await async_client.post(
        "/api/v1/query",
        json={
            "question": "Tell me about chapter 1",
            "document_ids": ["specific-doc-id"],
        },
    )
    assert response.status_code == 200
    # Ensure filter was passed through
    call_kwargs = mock_vector_store.search.call_args.kwargs
    assert call_kwargs["document_ids"] == ["specific-doc-id"]


@pytest.mark.asyncio
async def test_query_empty_question(async_client):
    response = await async_client.post(
        "/api/v1/query",
        json={"question": ""},
    )
    assert response.status_code == 422  # Pydantic min_length validation


@pytest.mark.asyncio
async def test_query_top_k_validation(async_client):
    response = await async_client.post(
        "/api/v1/query",
        json={"question": "hello", "top_k": 0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_response_schema(async_client, mock_vector_store, mock_llm):
    """Verify all expected fields are present in the response."""
    mock_vector_store.search = AsyncMock(
        return_value=[
            SearchResult("id1", "doc1", 0, "Some text.", 0.1, 2)
        ]
    )
    mock_llm.generate = AsyncMock(return_value="Answer.")
    response = await async_client.post(
        "/api/v1/query",
        json={"question": "Test question"},
    )
    data = response.json()
    assert set(data.keys()) >= {"answer", "sources", "question"}
    assert data["question"] == "Test question"
