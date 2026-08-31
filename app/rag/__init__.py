"""LexoraAI RAG engine.

Each stage of the pipeline lives in its own module:

    parser.py       PDF/DOCX/TXT/MD -> clean, structured blocks
    chunker.py      blocks -> semantically coherent, context-headed chunks
    embedder.py     text -> vectors (Jina v4, role-aware, cached, batched)
    vectorstore.py  dense vector index (ChromaDB)
    keyword.py      sparse lexical index (BM25)
    retriever.py    hybrid search: dense + sparse -> RRF -> MMR
    reranker.py     precision pass: LLM / cross-encoder / heuristic
    reasoning.py    query routing, rewriting, decomposition, HyDE
    generator.py    grounded, cited answer generation
    grounding.py    citation validation + faithfulness scoring
    llm.py          Groq client: retries, model fallback, circuit breaker
    prompts.py      every prompt template in one place
    pipeline.py     orchestrates ingestion and question answering
"""
