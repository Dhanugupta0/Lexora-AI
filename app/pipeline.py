import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_collection = None
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        os.makedirs(settings.CHROMA_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


@dataclass
class ParsedDoc:
    pages: List[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def parse_document(file_bytes: bytes, file_type: str) -> ParsedDoc:
    file_type = file_type.lower().lstrip(".")

    if file_type == "pdf":
        import fitz
        pages = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if text:
                    pages.append(text)
        if not pages:
            raise ValueError("PDF has no extractable text (may be a scanned image).")
        return ParsedDoc(pages=pages)

    elif file_type == "docx":
        import io
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        sections, current = [], []
        for para in doc.paragraphs:
            if para.text.strip():
                current.append(para.text.strip())
            elif current:
                sections.append("\n".join(current))
                current = []
        if current:
            sections.append("\n".join(current))
        if not sections:
            raise ValueError("DOCX has no extractable text.")
        return ParsedDoc(pages=sections)

    elif file_type == "txt":
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        if not text.strip():
            raise ValueError("Text file is empty.")
        lines = text.splitlines()
        pages = [
            "\n".join(lines[i : i + 50]).strip()
            for i in range(0, len(lines), 50)
            if "\n".join(lines[i : i + 50]).strip()
        ]
        return ParsedDoc(pages=pages)

    else:
        raise ValueError(f"Unsupported type '{file_type}'. Allowed: pdf, docx, txt")


@dataclass
class Chunk:
    text: str
    token_count: int
    chunk_index: int
    page_number: int


def chunk_text(
    pages: List[str],
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[Chunk]:
    chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
    overlap = overlap or settings.CHUNK_OVERLAP_TOKENS

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    enc = tiktoken.get_encoding("cl100k_base")

    all_tokens: List[int] = []
    token_page: List[int] = []
    for page_num, page_text in enumerate(pages, start=1):
        tokens = enc.encode(page_text)
        all_tokens.extend(tokens)
        token_page.extend([page_num] * len(tokens))

    if not all_tokens:
        return []

    stride = chunk_size - overlap
    chunks: List[Chunk] = []

    start = 0
    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        token_slice = all_tokens[start:end]
        chunks.append(Chunk(
            text=enc.decode(token_slice),
            token_count=len(token_slice),
            chunk_index=len(chunks),
            page_number=token_page[start],
        ))
        if end == len(all_tokens):
            break
        start += stride

    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def store_chunks(
    document_id: str,
    chunks: List[Chunk],
    embeddings: List[List[float]],
) -> List[str]:
    collection = get_collection()
    ids = [f"{document_id}_{c.chunk_index}" for c in chunks]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"document_id": document_id, "chunk_index": c.chunk_index, "page_number": c.page_number}
            for c in chunks
        ],
    )
    return ids


def delete_document_vectors(document_id: str) -> None:
    collection = get_collection()
    collection.delete(where={"document_id": document_id})


@dataclass
class SearchResult:
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    score: float


def search_chunks(
    question: str,
    top_k: int = 5,
    document_ids: Optional[List[str]] = None,
) -> List[SearchResult]:
    collection = get_collection()

    total = collection.count()
    if total == 0:
        return []

    [query_vector] = get_embeddings([question])

    where = None
    if document_ids:
        where = (
            {"document_id": document_ids[0]}
            if len(document_ids) == 1
            else {"document_id": {"$in": document_ids}}
        )

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results: List[SearchResult] = []
    for chroma_id, text, meta, distance in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(SearchResult(
            document_id=meta["document_id"],
            chunk_index=meta["chunk_index"],
            page_number=meta.get("page_number", 0),
            text=text,
            score=round(1 - distance, 4),
        ))

    return results


def generate_answer(question: str, context_chunks: List[SearchResult]) -> str:
    from openai import OpenAI

    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[Source {i} — Document: {chunk.document_id}, Page: {chunk.page_number}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are LexoraAI, a document assistant. "
        "Answer the question using ONLY the context excerpts below. "
        "If the answer is not in the context, say so honestly. "
        "Do not make up information."
    )
    user_prompt = f"Context:\n\n{context}\n\n---\nQuestion: {question}\n\nAnswer:"

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
