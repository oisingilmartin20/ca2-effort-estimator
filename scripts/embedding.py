"""Shared embedding helpers for batch ingestion and MCP similarity search."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

Provider = Literal["local", "openai"]

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_DIMENSION = 384
DEFAULT_OPENAI_DIMENSION = 1536


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Provider
    model: str
    dimension: int


def _parse_provider(value: str | None) -> Provider:
    provider = (value or "local").strip().lower()
    if provider not in {"local", "openai"}:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {value!r}. Use 'local' or 'openai'.")
    return provider  # type: ignore[return-value]


def get_embedding_config() -> EmbeddingConfig:
    """Read embedding settings from environment variables."""
    load_dotenv()
    provider = _parse_provider(os.getenv("EMBEDDING_PROVIDER"))

    if provider == "local":
        model = os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL)
        dimension = int(os.getenv("EMBEDDING_DIMENSION", DEFAULT_LOCAL_DIMENSION))
    else:
        model = os.getenv("EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL)
        dimension = int(os.getenv("EMBEDDING_DIMENSION", DEFAULT_OPENAI_DIMENSION))

    if dimension <= 0:
        raise ValueError(f"EMBEDDING_DIMENSION must be positive, got {dimension}")

    return EmbeddingConfig(provider=provider, model=model, dimension=dimension)


@lru_cache(maxsize=1)
def _get_local_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    model = _get_local_model(model_name)
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def _embed_openai(texts: list[str], model_name: str) -> list[list[float]]:
    client = _get_openai_client()
    response = client.embeddings.create(model=model_name, input=texts)
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str], config: EmbeddingConfig | None = None) -> list[list[float]]:
    """Embed a batch of texts using the configured provider."""
    if not texts:
        return []

    cfg = config or get_embedding_config()
    if cfg.provider == "local":
        vectors = _embed_local(texts, cfg.model)
    else:
        vectors = _embed_openai(texts, cfg.model)

    for vector in vectors:
        if len(vector) != cfg.dimension:
            raise ValueError(
                f"Embedding dimension mismatch for model {cfg.model!r}: "
                f"expected {cfg.dimension}, got {len(vector)}"
            )
    return vectors


def embed_query(text: str, config: EmbeddingConfig | None = None) -> list[float]:
    """Embed a single query string."""
    vectors = embed_texts([text], config=config)
    return vectors[0]


def vector_to_pg_literal(vector: list[float]) -> str:
    """Format a float vector for pgvector SQL literals."""
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
