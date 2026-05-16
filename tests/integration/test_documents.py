"""Integration tests for document management endpoints."""
import pytest


@pytest.mark.asyncio
async def test_list_documents_empty(async_client):
    response = await async_client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_get_document_not_found(async_client):
    response = await async_client.get("/api/v1/documents/non-existent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_not_found(async_client):
    response = await async_client.delete("/api/v1/documents/non-existent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_documents_pagination_params(async_client):
    response = await async_client.get("/api/v1/documents?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 10


@pytest.mark.asyncio
async def test_list_documents_invalid_page(async_client):
    response = await async_client.get("/api/v1/documents?page=0")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert data["app"] == "LexoraAI"


@pytest.mark.asyncio
async def test_openapi_docs_available(async_client):
    response = await async_client.get("/docs")
    assert response.status_code == 200
