import io
import pytest
from unittest.mock import MagicMock, patch


class TestParser:
    def test_parse_txt_basic(self):
        from app.pipeline import parse_document
        result = parse_document(b"Hello world\nLine two", "txt")
        assert result.page_count >= 1
        assert "Hello world" in result.pages[0]

    def test_parse_txt_empty_raises(self):
        from app.pipeline import parse_document
        with pytest.raises(ValueError, match="empty"):
            parse_document(b"   ", "txt")

    def test_parse_unsupported_raises(self):
        from app.pipeline import parse_document
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document(b"data", "xlsx")

    def test_parse_txt_makes_pages(self):
        from app.pipeline import parse_document
        lines = "\n".join(f"Line {i}" for i in range(110))
        result = parse_document(lines.encode(), "txt")
        assert result.page_count >= 2

    def test_parse_txt_latin1_encoding(self):
        from app.pipeline import parse_document
        data = "Ça va".encode("latin-1")
        result = parse_document(data, "txt")
        assert result.page_count >= 1


class TestChunker:
    def test_returns_chunks(self):
        from app.pipeline import chunk_text
        chunks = chunk_text(["Hello world " * 50])
        assert len(chunks) >= 1

    def test_chunk_indices_sequential(self):
        from app.pipeline import chunk_text
        pages = ["word " * 300] * 3
        chunks = chunk_text(pages, chunk_size=128, overlap=16)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_chunk_size_respected(self):
        from app.pipeline import chunk_text
        pages = ["word " * 500]
        chunks = chunk_text(pages, chunk_size=64, overlap=8)
        for c in chunks:
            assert c.token_count <= 64

    def test_overlap_increases_chunk_count(self):
        from app.pipeline import chunk_text
        pages = ["word " * 500]
        without = chunk_text(pages, chunk_size=128, overlap=0)
        with_overlap = chunk_text(pages, chunk_size=128, overlap=64)
        assert len(with_overlap) >= len(without)

    def test_empty_pages_returns_empty(self):
        from app.pipeline import chunk_text
        assert chunk_text([]) == []

    def test_invalid_overlap_raises(self):
        from app.pipeline import chunk_text
        with pytest.raises(ValueError):
            chunk_text(["hello"], chunk_size=50, overlap=50)

    def test_page_number_tracked(self):
        from app.pipeline import chunk_text
        chunks = chunk_text(["Page one " * 50, "Page two " * 50], chunk_size=64, overlap=0)
        assert chunks[0].page_number == 1


class TestEmbedder:
    def test_empty_input_returns_empty(self):
        from app.pipeline import get_embeddings
        assert get_embeddings([]) == []

    @patch("app.pipeline.OpenAI")
    def test_embedder_returns_vectors(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = MagicMock(data=[mock_item])

        from app.pipeline import get_embeddings
        results = get_embeddings(["hello world"])
        assert len(results) == 1
        assert isinstance(results[0], list)
        assert len(results[0]) > 0

    @patch("app.pipeline.OpenAI")
    def test_embedder_batch(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_item_1 = MagicMock()
        mock_item_1.embedding = [0.1, 0.2, 0.3]
        mock_item_2 = MagicMock()
        mock_item_2.embedding = [0.4, 0.5, 0.6]
        mock_client.embeddings.create.return_value = MagicMock(data=[mock_item_1, mock_item_2])

        from app.pipeline import get_embeddings
        results = get_embeddings(["first sentence", "second sentence"])
        assert len(results) == 2
        assert results[0] != results[1]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    import app.database as db_module

    with patch.object(db_module, "init_db", return_value=None), \
         patch.object(db_module, "get_db", return_value=iter([MagicMock()])):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestAPI:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_upload_unsupported_type(self, client):
        r = client.post(
            "/api/v1/upload",
            files=[("files", ("test.xlsx", io.BytesIO(b"data"), "application/octet-stream"))],
        )
        assert r.status_code == 415

    def test_upload_too_many_files(self, client):
        files = [("files", (f"f{i}.txt", io.BytesIO(b"hi"), "text/plain")) for i in range(21)]
        r = client.post("/api/v1/upload", files=files)
        assert r.status_code == 400

    def test_query_empty_question(self, client):
        r = client.post("/api/v1/query", json={"question": ""})
        assert r.status_code == 422

    def test_get_nonexistent_document(self, client):
        r = client.get("/api/v1/documents/does-not-exist")
        assert r.status_code in (404, 500)
