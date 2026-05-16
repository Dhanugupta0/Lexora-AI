"""Integration tests for POST /api/v1/upload."""
import io

import pytest


@pytest.mark.asyncio
async def test_upload_single_txt(async_client, tmp_path):
    content = b"This is a test document for upload testing."
    response = await async_client.post(
        "/api/v1/upload",
        files=[("files", ("test.txt", io.BytesIO(content), "text/plain"))],
    )
    assert response.status_code == 202
    data = response.json()
    assert "accepted" in data
    assert len(data["accepted"]) == 1
    assert "document_id" in data["accepted"][0]


@pytest.mark.asyncio
async def test_upload_unsupported_type(async_client):
    response = await async_client.post(
        "/api/v1/upload",
        files=[("files", ("data.xlsx", io.BytesIO(b"data"), "application/octet-stream"))],
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_upload_no_files(async_client):
    response = await async_client.post("/api/v1/upload", files=[])
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_upload_too_many_files(async_client):
    files = [
        ("files", (f"doc{i}.txt", io.BytesIO(b"content"), "text/plain"))
        for i in range(21)
    ]
    response = await async_client.post("/api/v1/upload", files=files)
    assert response.status_code == 400
    assert "Max" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_multiple_files(async_client):
    files = [
        ("files", (f"doc{i}.txt", io.BytesIO(f"Content {i}".encode()), "text/plain"))
        for i in range(3)
    ]
    response = await async_client.post("/api/v1/upload", files=files)
    assert response.status_code == 202
    assert len(response.json()["accepted"]) == 3


@pytest.mark.asyncio
async def test_upload_returns_document_ids(async_client):
    content = b"Hello, document!"
    response = await async_client.post(
        "/api/v1/upload",
        files=[("files", ("hello.txt", io.BytesIO(content), "text/plain"))],
    )
    accepted = response.json()["accepted"]
    doc_id = accepted[0]["document_id"]
    assert isinstance(doc_id, str)
    assert len(doc_id) == 36  # UUID4 format
