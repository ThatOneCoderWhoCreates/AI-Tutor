"""Tests run fully offline: mock LLM, hash embeddings, a temporary copy of the demo data."""
import os
import shutil
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="tutor_test_"))
shutil.copytree(Path(__file__).resolve().parent.parent / "data" / "lectures" / "demo", _tmp / "lectures" / "demo")

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["VECTOR_STORE"] = os.environ.get("TEST_VECTOR_STORE", "memory")
os.environ["DATA_DIR"] = str(_tmp)
