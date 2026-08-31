import io
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestSystemEndpoints:
    def test_health(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok" and body["app"] == "LexoraAI"

    def test_openapi_docs_render(self, client):
        assert client.get("/docs").status_code == 200

    def test_stats_reports_the_corpus(self, client):
        body = client.get("/api/v1/stats").json()
        assert "documents" in body and "chunks" in body and "pipeline" in body

    def test_stats_exposes_the_pipeline_configuration(self, client):
        config = client.get("/api/v1/stats").json()["pipeline"]["config"]
        assert config["retrieval_mode"] in ("hybrid", "dense", "sparse")
        assert "chunk_strategy" in config


class TestUploadValidation:
    def test_rejects_unsupported_types(self, client):
        r = client.post("/api/v1/upload",
                        files=[("files", ("a.xlsx", io.BytesIO(b"data"), "application/octet-stream"))])
        assert r.status_code == 415

    def test_rejects_too_many_files(self, client):
        files = [("files", (f"f{i}.txt", io.BytesIO(b"hi"), "text/plain")) for i in range(21)]
        assert client.post("/api/v1/upload", files=files).status_code == 400

    def test_rejects_empty_files(self, client):
        r = client.post("/api/v1/upload",
                        files=[("files", ("a.txt", io.BytesIO(b""), "text/plain"))])
        assert r.status_code == 400

    def test_accepts_markdown(self, client):
        with patch("app.api.upload.ingest_document") as ingest:
            ingest.return_value = MagicMock(title="T", page_count=1, chunk_count=1,
                                            token_count=10, seconds=0.1)
            r = client.post("/api/v1/upload",
                            files=[("files", ("notes.md", io.BytesIO(b"# Title\nBody"), "text/markdown"))])
        assert r.status_code == 202 and r.json()["count"] == 1


class TestQueryEndpoint:
    def test_rejects_an_empty_question(self, client):
        assert client.post("/api/v1/query", json={"question": ""}).status_code == 422

    def test_rejects_an_out_of_range_top_k(self, client):
        assert client.post("/api/v1/query",
                           json={"question": "hi", "top_k": 99}).status_code == 422

    def test_returns_the_full_grounding_envelope(self, client):
        from app.rag.pipeline import AnswerResult
        stub = AnswerResult(
            question="q", answer="a [S1]", sources=[], intent="document_qa",
            confidence=0.87, grounded=True, abstained=False, degraded=False,
            model="openai/gpt-oss-20b",
        )
        with patch("app.api.query.answer_question", return_value=stub):
            body = client.post("/api/v1/query", json={"question": "q"}).json()
        assert body["confidence"] == 0.87
        assert body["grounded"] is True and body["abstained"] is False
        assert body["model"] == "openai/gpt-oss-20b"

    def test_accepts_conversation_history(self, client):
        from app.rag.pipeline import AnswerResult
        stub = AnswerResult(question="q", answer="a")
        with patch("app.api.query.answer_question", return_value=stub) as answer:
            r = client.post("/api/v1/query", json={
                "question": "and when is it due?",
                "history": [{"role": "user", "content": "What is NW-4417?"},
                            {"role": "assistant", "content": "A renewal contract."}],
            })
        assert r.status_code == 200
        assert len(answer.call_args.kwargs["history"]) == 2

    def test_rejects_an_invalid_history_role(self, client):
        r = client.post("/api/v1/query", json={
            "question": "q", "history": [{"role": "system", "content": "x"}]})
        assert r.status_code == 422


class TestDocumentEndpoints:
    def test_missing_document_returns_404(self, client):
        assert client.get("/api/v1/documents/does-not-exist").status_code == 404

    def test_deleting_a_missing_document_returns_404(self, client):
        assert client.delete("/api/v1/documents/does-not-exist").status_code == 404

    def test_listing_documents_returns_a_list(self, client):
        r = client.get("/api/v1/documents")
        assert r.status_code == 200 and isinstance(r.json(), list)

    def test_rejects_an_invalid_status_filter(self, client):
        assert client.get("/api/v1/documents?status=bogus").status_code == 422
