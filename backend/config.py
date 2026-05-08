# -*- coding: utf-8 -*-
"""
Centralized configuration via environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "Agentic Graph RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Neo4j ────────────────────────────────────────────
    NEO4J_URI: str = "neo4j+s://e742217b.databases.neo4j.io"
    NEO4J_USER: str = "e742217b"
    NEO4J_PASSWORD: str = "AGTXuaLS8hYDT10a5pkOM3km7bjCC2QDzxPULi_Rngg"

    # ── FAISS / Embeddings ───────────────────────────────
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    HYBRID_VECTOR_WEIGHT: float = 0.1
    GRAPH_STORE_PATH: str = "data/graph_store.json"
    GRAPH_NER_MODEL: str = "Jean-Baptiste/camembert-ner"

    # ── Corpus ───────────────────────────────────────────
    CORPUS_DIR: str = "data/corpus"

    # ── LLM ──────────────────────────────────────────────
    LLM_PROVIDER: str = "groq" # "gemini" or "groq"
    
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
