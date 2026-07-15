"""Batch-embed the retrieval corpus CSV into Postgres pgvector.

Run:  python scripts/generate_embeddings.py

Requires Postgres running (docker compose up -d) and the retrieval CSV created
by scripts/create_train_retrieval_split.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from embedding import EmbeddingConfig, embed_texts, get_embedding_config, vector_to_pg_literal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "tawos_retrieval_corpus.csv"
DEFAULT_POSTGRES_URL = "postgresql://tawos:tawos@localhost:5432/tawos_vectors"

UPSERT_SQL = """
INSERT INTO ticket_embeddings (
  issue_key, title, description, story_points, embedding, embedding_model
)
VALUES (
  %(issue_key)s, %(title)s, %(description)s, %(story_points)s,
  %(embedding)s::vector, %(embedding_model)s
)
ON CONFLICT (issue_key) DO UPDATE SET
  title = EXCLUDED.title,
  description = EXCLUDED.description,
  story_points = EXCLUDED.story_points,
  embedding = EXCLUDED.embedding,
  embedding_model = EXCLUDED.embedding_model,
  created_at = NOW()
"""


def _normalize_postgres_url(url: str) -> str:
    """Convert SQLAlchemy-style Postgres URLs to psycopg-compatible URLs."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def _embedding_text(row: pd.Series, *, include_title: bool) -> str:
    description = str(row.get("description", "")).strip()
    if not include_title:
        return description
    title = str(row.get("title", "")).strip()
    if title:
        return f"{title}\n\n{description}"
    return description


def _existing_issue_keys(
    conn: psycopg.Connection,
    model_name: str,
) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT issue_key FROM ticket_embeddings WHERE embedding_model = %s",
            (model_name,),
        )
        return {row[0] for row in cur.fetchall()}


def ingest_csv(
    csv_path: Path,
    postgres_url: str,
    *,
    config: EmbeddingConfig,
    batch_size: int = 128,
    force: bool = False,
    limit: int | None = None,
    include_title: bool = False,
) -> int:
    """Embed rows from csv_path and upsert them into ticket_embeddings."""
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    if df.empty:
        return 0

    postgres_url = _normalize_postgres_url(postgres_url)
    written = 0

    with psycopg.connect(postgres_url) as conn:
        existing = set() if force else _existing_issue_keys(conn, config.model)
        pending_rows: list[pd.Series] = []

        for _, row in df.iterrows():
            issue_key = str(row.get("issue_key", "")).strip()
            if not issue_key:
                continue
            if issue_key in existing:
                continue
            pending_rows.append(row)

        for start in range(0, len(pending_rows), batch_size):
            batch = pending_rows[start : start + batch_size]
            texts = [_embedding_text(row, include_title=include_title) for row in batch]
            vectors = embed_texts(texts, config=config)

            with conn.cursor() as cur:
                for row, vector in zip(batch, vectors):
                    story_points = int(row["actual_story_points"])
                    params = {
                        "issue_key": str(row["issue_key"]),
                        "title": str(row.get("title", "")).strip() or None,
                        "description": str(row.get("description", "")).strip(),
                        "story_points": story_points,
                        "embedding": vector_to_pg_literal(vector),
                        "embedding_model": config.model,
                    }
                    cur.execute(UPSERT_SQL, params)
                    written += 1
            conn.commit()
            print(f"Upserted {min(start + batch_size, len(pending_rows)):,}/{len(pending_rows):,} rows")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed retrieval corpus tickets into Postgres.")
    parser.add_argument("--csv", type=Path, default=None, help="Retrieval corpus CSV path")
    parser.add_argument("--postgres-url", default=None, help="Postgres connection URL")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    parser.add_argument("--force", action="store_true", help="Re-embed rows for the current model")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Prepend title to the embedded text (default: description only)",
    )
    args = parser.parse_args()

    load_dotenv()
    csv_path = args.csv or Path(os.getenv("TAWOS_RETRIEVAL_CSV", DEFAULT_CSV))
    postgres_url = args.postgres_url or os.getenv("POSTGRES_URL", DEFAULT_POSTGRES_URL)
    config = get_embedding_config()

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}", file=sys.stderr)
        print(
            "Run scripts/create_train_retrieval_split.py first, or pass --csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        written = ingest_csv(
            csv_path,
            postgres_url,
            config=config,
            batch_size=args.batch_size,
            force=args.force,
            limit=args.limit,
            include_title=args.include_title,
        )
    except psycopg.Error as exc:
        print("Error: could not connect to Postgres or upsert embeddings.", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print(
            "\nEnsure Postgres is running:\n"
            "  docker compose up -d\n"
            f"Connection string: {postgres_url}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Embedding ingestion complete ===")
    print(f"Provider / model:  {config.provider} / {config.model}")
    print(f"Source CSV:        {csv_path}")
    print(f"Rows upserted:     {written:,}")


if __name__ == "__main__":
    main()
