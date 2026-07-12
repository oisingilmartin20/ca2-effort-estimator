"""MCP server for TAWOS ticket similarity search via Postgres pgvector."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from similarity_search import find_similar_tickets as search_similar_tickets  # noqa: E402

mcp = FastMCP("TAWOS Similarity")


@mcp.tool
def find_similar_tickets(description: str, limit: int = 10) -> str:
    """Find tickets with descriptions most similar to the query.

    Embeds the query description, searches Postgres pgvector for nearest
    neighbours, and returns the closest tickets with description and story points.
    """
    if not description.strip():
        return json.dumps({"error": "description must not be empty"})

    try:
        results = search_similar_tickets(description, limit=limit)
    except Exception as exc:
        return json.dumps({"error": f"similarity search failed: {exc}"})

    return json.dumps([asdict(ticket) for ticket in results], indent=2)


if __name__ == "__main__":
    mcp.run()
