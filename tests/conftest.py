import os
import tempfile

# Point every storage path at a throwaway directory before app modules import.
_TMP = tempfile.mkdtemp(prefix="lexora-tests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("CHROMA_PATH", f"{_TMP}/chroma")
os.environ.setdefault("UPLOAD_DIR", f"{_TMP}/uploads")
os.environ.setdefault("GROQ_API_KEY", "")          # offline by default
os.environ.setdefault("JINA_API_KEY", "")          # never call the embeddings API in tests

import pytest


@pytest.fixture(scope="session")
def tmp_root():
    return _TMP
