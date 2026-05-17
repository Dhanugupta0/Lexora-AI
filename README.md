# ⚡ LexoraAI

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets users upload documents, ask natural-language questions, and receive grounded answers backed by source citations.

Built with **FastAPI**, **ChromaDB**, **Sentence-Transformers**, and **OpenAI GPT** — deployable locally or via Docker in minutes.

---

## Table of Contents

- [What is RAG?](#what-is-rag)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [How the Pipeline Works](#how-the-pipeline-works)
- [API Endpoints](#api-endpoints)
- [Getting Started](#getting-started)
  - [Local Setup](#local-setup)
  - [Docker Deployment](#docker-deployment)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Diagrams](#diagrams)

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that improves LLM answers by first **retrieving** relevant content from a knowledge base, then **generating** a response grounded in that content. This eliminates hallucination because the LLM can only answer from the documents you provide.

```
Traditional LLM:  Question → LLM → Answer (may hallucinate)
RAG Pipeline:     Question → Search Documents → LLM + Context → Grounded Answer ✓
```

---

## Features

- **Multi-format document upload** — PDF, DOCX, and TXT files up to 50 MB each
- **Automatic text extraction** — PyMuPDF for PDFs, python-docx for DOCX, built-in for TXT
- **Token-aware chunking** — sliding window chunker with configurable size and overlap
- **Local embeddings** — Sentence-Transformers (all-MiniLM-L6-v2), no API key needed for embedding
- **Vector search** — ChromaDB with cosine similarity for fast semantic retrieval
- **LLM-powered answers** — OpenAI GPT generates answers using only retrieved context
- **Source citations** — every answer includes the exact chunks and pages used
- **Background processing** — documents are ingested asynchronously after upload
- **Gradio frontend** — clean chat UI with document management
- **REST API** — full Swagger/OpenAPI documentation at `/docs`
- **Docker-ready** — single `docker-compose up` for production deployment
- **21 unit + integration tests** — all passing

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI | REST API with async support |
| Database | SQLite (local) / PostgreSQL (Docker) | Document metadata storage |
| Vector Store | ChromaDB (embedded) | Stores and searches chunk embeddings |
| Embeddings | Sentence-Transformers | Converts text to vectors locally |
| LLM | OpenAI GPT-4o-mini | Generates answers from retrieved context |
| Document Parsing | PyMuPDF, python-docx | Extracts text from PDF and DOCX |
| Tokenizer | tiktoken | Token-accurate chunk splitting |
| Frontend | Gradio | Chat UI + document management |
| Testing | pytest | Unit and integration tests |

---

## Project Structure

```
LexoraAI/
├── app/
│   ├── __init__.py
│   ├── config.py            # Settings loaded from .env
│   ├── database.py          # SQLAlchemy models + engine (SQLite/PostgreSQL)
│   ├── main.py              # FastAPI app, CORS, lifespan startup
│   ├── pipeline.py          # Core RAG pipeline (parse → chunk → embed → store → search → generate)
│   └── api/
│       ├── __init__.py
│       ├── router.py        # Mounts all route modules under /api/v1
│       ├── upload.py        # POST /upload — file ingestion endpoint
│       ├── query.py         # POST /query — RAG question-answering endpoint
│       └── documents.py     # GET/DELETE /documents — document management
├── tests/
│   └── test_pipeline.py     # 21 tests covering parser, chunker, embedder, and API
├── frontend.py              # Gradio UI (chat + document management)
├── Dockerfile               # Container build for the API
├── docker-compose.yml        # PostgreSQL + API orchestration
├── requirements.txt          # Python dependencies
├── pytest.ini                # Test configuration
├── .env.example              # Template for environment variables
├── .gitignore
├── postman_collection.json   # Ready-to-import Postman collection
└── README.md
```

---

## Architecture

```mermaid
graph TB
    subgraph "Frontend"
        GR["Gradio UI<br/>(port 7860)"]
    end

    subgraph "API Layer"
        FA["FastAPI<br/>(port 8000)"]
        UP["/upload"]
        QR["/query"]
        DC["/documents"]
        HL["/health"]
    end

    subgraph "Pipeline"
        PA["Parser<br/>(PyMuPDF, python-docx)"]
        CH["Chunker<br/>(tiktoken, sliding window)"]
        EM["Embedder<br/>(Sentence-Transformers)"]
        GEN["Generator<br/>(OpenAI GPT)"]
    end

    subgraph "Storage"
        DB["SQLite / PostgreSQL<br/>(document metadata)"]
        CR["ChromaDB<br/>(vector embeddings)"]
        FS["File System<br/>(raw uploads)"]
    end

    GR -->|HTTP| FA
    FA --> UP
    FA --> QR
    FA --> DC
    FA --> HL

    UP --> PA --> CH --> EM --> CR
    UP --> DB
    UP --> FS

    QR --> EM
    QR --> CR
    QR --> GEN
    QR --> DB

    DC --> DB
    DC --> CR
    DC --> FS
```

---

## How the Pipeline Works

### Document Ingestion (Upload)

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant DB as Database
    participant P as Parser
    participant C as Chunker
    participant E as Embedder
    participant V as ChromaDB

    U->>API: POST /upload (file)
    API->>DB: Create document record (status: pending)
    API->>U: 202 Accepted

    Note over API: Background task starts

    API->>P: parse_document(bytes, type)
    P-->>API: ParsedDoc (list of pages)

    API->>C: chunk_text(pages)
    C-->>API: List of Chunks (512 tokens, 50 overlap)

    API->>E: get_embeddings(chunk texts)
    E-->>API: List of float vectors (384-dim)

    API->>V: store_chunks(doc_id, chunks, embeddings)
    API->>DB: Update status → ready, set chunk_count
```

### Question Answering (Query)

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant E as Embedder
    participant V as ChromaDB
    participant LLM as OpenAI GPT

    U->>API: POST /query {question, top_k}

    API->>E: get_embeddings([question])
    E-->>API: Query vector (384-dim)

    API->>V: query(vector, top_k)
    V-->>API: Top-K similar chunks + scores

    API->>LLM: Generate answer with context
    Note over LLM: System prompt enforces<br/>answering ONLY from context

    LLM-->>API: Grounded answer

    API->>U: {answer, sources[]}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/upload` | Upload 1–20 documents (PDF, DOCX, TXT) |
| `POST` | `/api/v1/query` | Ask a question, get a RAG-powered answer |
| `GET` | `/api/v1/documents` | List all documents with status |
| `GET` | `/api/v1/documents/{id}` | Get single document metadata |
| `DELETE` | `/api/v1/documents/{id}` | Delete document from DB, ChromaDB, and disk |

Full interactive docs available at `http://localhost:8000/docs` when the server is running.

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key (for answer generation only — embeddings run locally)

### Local Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd LexoraAI

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 5. Start the API server
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. (Optional) Start the Gradio frontend in a second terminal
PYTHONPATH=. python frontend.py
```

**Access:**
- API Swagger UI: http://localhost:8000/docs
- Gradio Frontend: http://localhost:7860
- Health Check: http://localhost:8000/api/v1/health

### Docker Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 2. Build and start (PostgreSQL + API)
docker-compose up --build -d

# 3. Check logs
docker-compose logs -f api
```

The Docker setup uses **PostgreSQL** for metadata and **ChromaDB embedded** for vectors. Both persist data via Docker volumes.

---

## Running Tests

```bash
# Run all 21 tests
PYTHONPATH=. pytest tests/ -v

# Expected output:
# 21 passed, 0 failed
```

Tests cover:
- **Parser** — TXT parsing, encoding fallback, empty file handling, unsupported types
- **Chunker** — chunk sizes, overlap, sequential indices, page tracking
- **Embedder** — empty input, vector output, batch processing
- **API** — health check, upload validation, query validation, document retrieval

---

## Configuration

All settings are loaded from `.env` (see `.env.example` for the template):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./local_data/lexora.db` | Database connection string |
| `CHROMA_PATH` | `./local_data/chroma` | ChromaDB storage directory |
| `UPLOAD_DIR` | `./local_data/uploads` | Uploaded files directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model name |
| `OPENAI_API_KEY` | — | Required for answer generation |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for answer generation |
| `CHUNK_SIZE_TOKENS` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | `5` | Default number of chunks to retrieve |
| `MAX_FILE_SIZE_MB` | `50` | Max upload file size |
| `MAX_DOCUMENTS` | `20` | Max files per upload request |

---

## Diagrams

### Data Flow Overview

```mermaid
flowchart LR
    subgraph INPUT
        PDF["📄 PDF"]
        DOCX["📝 DOCX"]
        TXT["📃 TXT"]
    end

    subgraph INGESTION["Ingestion Pipeline"]
        PARSE["Parse<br/>Extract text"]
        CHUNK["Chunk<br/>512 tokens<br/>50 overlap"]
        EMBED["Embed<br/>all-MiniLM-L6-v2<br/>384 dimensions"]
        STORE["Store<br/>ChromaDB"]
    end

    subgraph QUERY["Query Pipeline"]
        QEMBED["Embed Question"]
        SEARCH["Vector Search<br/>Cosine Similarity"]
        CONTEXT["Build Context<br/>Top-K chunks"]
        GENERATE["Generate Answer<br/>OpenAI GPT"]
    end

    subgraph OUTPUT
        ANS["💬 Answer"]
        SRC["📚 Sources"]
    end

    PDF --> PARSE
    DOCX --> PARSE
    TXT --> PARSE
    PARSE --> CHUNK --> EMBED --> STORE

    QEMBED --> SEARCH --> CONTEXT --> GENERATE
    STORE -.->|similarity search| SEARCH
    GENERATE --> ANS
    SEARCH --> SRC
```

### Component Dependency Graph

```mermaid
graph TD
    MAIN["main.py<br/>FastAPI App"] --> CONFIG["config.py<br/>Settings"]
    MAIN --> DATABASE["database.py<br/>SQLAlchemy"]
    MAIN --> ROUTER["router.py<br/>API Router"]

    ROUTER --> UPLOAD["upload.py<br/>File Ingestion"]
    ROUTER --> QUERYAPI["query.py<br/>Question Answering"]
    ROUTER --> DOCS["documents.py<br/>Document Management"]

    UPLOAD --> PIPELINE["pipeline.py<br/>RAG Pipeline"]
    UPLOAD --> DATABASE
    QUERYAPI --> PIPELINE
    DOCS --> DATABASE
    DOCS --> PIPELINE

    PIPELINE --> CONFIG
    DATABASE --> CONFIG

    PIPELINE --> CHROMADB["ChromaDB"]
    PIPELINE --> STRANS["Sentence-Transformers"]
    PIPELINE --> OPENAI["OpenAI API"]
    PIPELINE --> PYMUPDF["PyMuPDF"]
    PIPELINE --> TIKTOKEN["tiktoken"]

    DATABASE --> SQLA["SQLAlchemy"]

    FRONTEND["frontend.py<br/>Gradio UI"] -->|HTTP| MAIN

    style MAIN fill:#4f46e5,color:#fff
    style PIPELINE fill:#7c3aed,color:#fff
    style FRONTEND fill:#059669,color:#fff
    style DATABASE fill:#d97706,color:#fff
```

### Database Schema

```mermaid
erDiagram
    DOCUMENTS {
        string id PK "UUID (36 chars)"
        string filename "Original filename"
        string file_type "pdf | docx | txt"
        bigint file_size "Size in bytes"
        int page_count "Pages extracted"
        int chunk_count "Chunks created"
        enum status "pending | processing | ready | error"
        text error_message "Error details if failed"
        datetime upload_time "When uploaded"
        datetime processed_time "When processing finished"
    }
```

### Document Status Lifecycle

```mermaid
stateDiagram-v2
    [*] --> pending: File uploaded
    pending --> processing: Background task starts
    processing --> ready: Ingestion complete
    processing --> error: Exception occurred
    ready --> [*]: Document available for queries
    error --> [*]: Check error_message field
    ready --> [*]: DELETE removes from all stores
```

### Chunking Strategy

```mermaid
graph LR
    subgraph "Document Text (tokenized)"
        T1["Token 1"]
        T2["Token 2"]
        T3["..."]
        T512["Token 512"]
        T513["Token 513"]
        T562["Token 562"]
        T563["Token 563"]
        T1024["Token 1024"]
    end

    subgraph "Chunk 0 (512 tokens)"
        C0["Tokens 1 → 512"]
    end

    subgraph "Chunk 1 (512 tokens, 50 overlap)"
        C1["Tokens 463 → 974"]
    end

    subgraph "Chunk 2 (remaining)"
        C2["Tokens 925 → 1024"]
    end

    T1 -.-> C0
    T512 -.-> C0
    T513 -.-> C1
    T1024 -.-> C2

    style C0 fill:#dbeafe
    style C1 fill:#e0e7ff
    style C2 fill:#ede9fe
```

### Deployment Options

```mermaid
graph TB
    subgraph "Local Development"
        L_PY["Python 3.10+"]
        L_UV["uvicorn (port 8000)"]
        L_GR["Gradio (port 7860)"]
        L_DB["SQLite"]
        L_CR["ChromaDB (./local_data)"]
        L_PY --> L_UV
        L_PY --> L_GR
        L_UV --> L_DB
        L_UV --> L_CR
    end

    subgraph "Docker Production"
        D_PG["PostgreSQL 16<br/>(Docker volume)"]
        D_API["FastAPI Container<br/>(port 8000)"]
        D_CR2["ChromaDB<br/>(Docker volume)"]
        D_GR2["Gradio<br/>(port 7860)"]
        D_API --> D_PG
        D_API --> D_CR2
        D_API --> D_GR2
    end

    style L_UV fill:#4f46e5,color:#fff
    style D_API fill:#4f46e5,color:#fff
    style L_DB fill:#d97706,color:#fff
    style D_PG fill:#d97706,color:#fff
```

---

**Built by Dhanu Gupta** — Internship project demonstrating a full-stack RAG pipeline from document ingestion to answer generation.
