"""Manual smoke test: start the MCP server over stdio and call every tool against the live API.

Usage:  python scripts/smoke_server.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastmcp import Client

SERVER = Path(__file__).resolve().parents[1] / "server" / "wb_docs_mcp.py"

CALLS = [
    ("get_facets", {"field": "document_type", "limit": 5}),
    ("get_facets", {"field": "language", "limit": 8}),
    ("resolve_filter_values", {"field": "sector", "terms": ["water and sanitation"]}),
    ("resolve_filter_values", {"field": "region", "terms": ["Sub-Saharan Africa"]}),
    ("search_documents", {"query": "climate resilience Kenya", "rows": 3}),
    ("filter_documents", {"country": "indonesia", "major_document_type": "Economic & Sector Work",
                          "rows": 3}),
    ("filter_documents", {"country": "Brazil", "sector": "water and sanitation",
                          "major_document_type": "Project Documents", "rows": 3}),
    ("filter_documents", {"region": "Sub-Saharan Africa", "query": "education financing",
                          "start_date": "2019-01-01", "end_date": "2022-12-31", "rows": 3}),
    ("filter_documents", {"country": "Kenia"}),  # expected: error with a suggestion
    ("get_document", {"document_id": "D33295056"}),
]


def summarize(name: str, data) -> str:
    if name == "get_document":
        return json.dumps({k: data.get(k) for k in ("id", "display_title", "docdt", "count", "docty")})
    if isinstance(data, dict) and "documents" in data:
        docs = [f"  - {d['date']} | {d['title'][:90]}" for d in data["documents"]]
        return (f"total={data['total']} next_offset={data['next_offset']} "
                f"applied={data.get('applied_filters')}\n" + "\n".join(docs))
    return json.dumps(data, ensure_ascii=False)[:500]


async def main() -> None:
    async with Client(SERVER) as client:
        tools = await client.list_tools()
        print("TOOLS:", ", ".join(t.name for t in tools))
        for name, args in CALLS:
            result = await client.call_tool(name, args, raise_on_error=False)
            print(f"\n>>> {name} {json.dumps(args)}")
            if result.is_error:
                print("ERROR:", " ".join(getattr(c, "text", "") for c in result.content))
            else:
                print(summarize(name, result.structured_content))


if __name__ == "__main__":
    asyncio.run(main())
