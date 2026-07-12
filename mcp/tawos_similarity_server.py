"""MCP server for TAWOS ticket similarity search via Postgres pgvector."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from embedding import embed_query, get_embedding_config, vector_to_pg_literal  # noqa: E402

DEFAULT_POSTGRES_URL = "postgresql://tawos:tawos@localhost:5432/tawos_vectors"

mcp = FastMCP("TAWOS Similarity")


def _normalize_postgres_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def _postgres_url() -> str:
    load_dotenv()
    return _normalize_postgres_url(os.getenv("POSTGRES_URL", DEFAULT_POSTGRES_URL))


@mcp.tool
def find_similar_tickets(description: str, limit: int = 10) -> str:
    """Find tickets with descriptions most similar to the query.

    Embeds the query description, searches Postgres pgvector for nearest
    neighbours, and returns the closest tickets with description and story points.
    """
    if not description.strip():
        return json.dumps({"error": "description must not be empty"})

    limit = max(1, min(limit, 50))
    config = get_embedding_config()
    query_vector = embed_query(description.strip(), config=config)
    vector_literal = vector_to_pg_literal(query_vector)

    query = """
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

    try:
        with psycopg.connect(_postgres_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (vector_literal, config.model, vector_literal, limit),
                )
                rows = cur.fetchall()
    except psycopg.Error as exc:
        return json.dumps({"error": f"database query failed: {exc}"})

    results = [
        {
            "issue_key": row[0],
            "title": row[1],
            "description": row[2],
            "story_points": row[3],
            "similarity": round(float(row[4]), 4),
        }
        for row in rows
    ]
    return json.dumps(results, indent=2)


if __name__ == "__main__":
    mcp.run()
