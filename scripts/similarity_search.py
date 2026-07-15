"""Postgres pgvector similarity search for TAWOS ticket descriptions."""
from __future__ import annotations

import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv

from embedding import embed_query, get_embedding_config, vector_to_pg_literal

DEFAULT_POSTGRES_URL = "postgresql://tawos:tawos@localhost:5432/tawos_vectors"

SIMILARITY_QUERY = """
    SELECT
      issue_key,
      title,
      description,
      story_points,
      1 - (embedding <=> %s::vector) AS similarity
    FROM ticket_embeddings
    WHERE embedding_model = %s
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""


@dataclass(frozen=True)
class SimilarTicket:
    issue_key: str
    title: str
    description: str
    story_points: int
    similarity: float


def normalize_postgres_url(url: str) -> str:
    """Convert SQLAlchemy-style Postgres URLs to psycopg-compatible URLs."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def postgres_url() -> str:
    load_dotenv()
    return normalize_postgres_url(os.getenv("POSTGRES_URL", DEFAULT_POSTGRES_URL))


def find_similar_tickets(description: str, limit: int = 10) -> list[SimilarTicket]:
    """Return nearest-neighbour tickets by description embedding similarity."""
    if not description.strip():
        return []

    limit = max(1, min(limit, 50))
    config = get_embedding_config()
    query_vector = embed_query(description.strip(), config=config)
    vector_literal = vector_to_pg_literal(query_vector)

    with psycopg.connect(postgres_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                SIMILARITY_QUERY,
                (vector_literal, config.model, vector_literal, limit),
            )
            rows = cur.fetchall()

    return [
        SimilarTicket(
            issue_key=row[0],
            title=row[1] or "",
            description=row[2],
            story_points=int(row[3]),
            similarity=round(float(row[4]), 4),
        )
        for row in rows
    ]
