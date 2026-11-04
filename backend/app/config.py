"""Single place for every setting.

All values come from backend/.env (copy .env.example to .env).
Switching models (for example qwen2.5:3b to qwen2.5:7b) means editing .env only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = _get("LLM_PROVIDER", "ollama")
    llm_model: str = _get("LLM_MODEL", "qwen2.5:7b")
    ollama_host: str = _get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    llm_num_ctx: int = int(_get("LLM_NUM_CTX", "8192"))
    llm_timeout: float = float(_get("LLM_TIMEOUT", "300"))
    # off = skip the "thinking" step of models like qwen3.5 (faster), on = let them think, auto = model default
    llm_think: str = _get("LLM_THINK", "off").lower()

    # Second model that answers questions the lectures do not cover (see app/fallback.py).
    # comma-separated chain, tried in order: gemini,groq,local  (or none)
    fallback_provider: str = _get("FALLBACK_PROVIDER", "none").lower()
    fallback_model: str = _get("FALLBACK_MODEL", "")
    fallback_api_key: str = _get("FALLBACK_API_KEY", "")
    fallback_timeout: float = float(_get("FALLBACK_TIMEOUT", "60"))
    gemini_api_key: str = _get("GEMINI_API_KEY", "")
    groq_api_key: str = _get("GROQ_API_KEY", "")
    openrouter_api_key: str = _get("OPENROUTER_API_KEY", "")
    gemini_model: str = _get("GEMINI_MODEL", "")
    groq_model: str = _get("GROQ_MODEL", "")
    openrouter_model: str = _get("OPENROUTER_MODEL", "")

    embedding_backend: str = _get("EMBEDDING_BACKEND", "sentence-transformers")
    embedding_model: str = _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    vector_store: str = _get("VECTOR_STORE", "chroma")

    whisper_backend: str = _get("WHISPER_BACKEND", "mlx")
    whisper_model: str = _get("WHISPER_MODEL", "small")
    scene_threshold: float = float(_get("SCENE_THRESHOLD", "0.01"))
    max_chunk_seconds: float = float(_get("MAX_CHUNK_SECONDS", "90"))

    data_dir: Path = Path(_get("DATA_DIR", str(BACKEND_DIR / "data"))).resolve()
    top_k: int = int(_get("TOP_K", "4"))
    cors_origins: str = _get("CORS_ORIGINS", "http://localhost:5173")

    @property
    def lectures_dir(self) -> Path:
        return self.data_dir / "lectures"

    @property
    def students_dir(self) -> Path:
        return self.data_dir / "students"

    @property
    def quizzes_dir(self) -> Path:
        return self.data_dir / "quizzes"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
