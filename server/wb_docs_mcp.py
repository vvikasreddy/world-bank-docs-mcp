"""MCP server exposing the World Bank Documents & Reports API as LLM tools (FastMCP, stdio).

Tools follow a discover -> resolve -> retrieve -> inspect workflow:

    get_facets             what values exist (document types, languages, countries, topics...)
    resolve_filter_values  map the user's wording to exact catalogue values ("kenya" -> "Kenya")
    filter_documents       structured retrieval (country/region/topic/sector/type/language/dates)
    search_documents       free-text retrieval
    get_document           full metadata for one document

Run:  python server/wb_docs_mcp.py          (stdio transport)
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Annotated, Any, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BeforeValidator, Field

if __package__ in (None, ""):  # executed as a script: make `server.*` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.wb_api import (  # noqa: E402
    FIELD_MAP,
    FacetField,
    FacetResult,
    ResolveResult,
    SearchResult,
    WorldBankAPIError,
    WorldBankClient,
)

WORKFLOW = """\
Tools for searching World Bank publications, research and project documents.
Recommended workflow:
1. DISCOVER - get_facets: answer "what types / languages / topics exist" questions and see the
   most common values of a field.
2. RESOLVE  - resolve_filter_values: turn the user's words into exact catalogue values before
   filtering (filters are exact-match and case-sensitive).
3. RETRIEVE - filter_documents for country/region/topic/sector/type/language/date questions;
   search_documents for free-text keyword questions.
4. INSPECT  - get_document for the full record of a specific document id.
Cite document titles, dates and URLs from tool results; never invent documents."""

mcp = FastMCP("worldbank-docs", instructions=WORKFLOW)
api = WorldBankClient()

T = TypeVar("T")


def _as_list(value: Any) -> Any:
    """Accept a single string where a list of strings is expected (small LLMs often do this)."""
    return [value] if isinstance(value, str) else value


# Shared, validated parameter types (FastMCP turns these into a Pydantic model + JSON schema).
Rows = Annotated[int, Field(ge=1, le=20, description="Results per page (1-20).")]
Offset = Annotated[int, Field(ge=0, description="Results to skip; use `next_offset` from a previous page.")]
IsoDate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date as YYYY-MM-DD.")]
FilterValues = Annotated[
    list[str] | None,
    BeforeValidator(_as_list),
    Field(max_length=5, description="One value or a list of values (combined with OR). "
                                    "Case-insensitive; must name a real catalogue value."),
]


async def _api(call: Awaitable[T]) -> T:
    """Run a World Bank API call, turning its failures into MCP tool errors the LLM can read."""
    try:
        return await call
    except WorldBankAPIError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def get_facets(
    field: Annotated[FacetField, Field(description="Which field to list values for.")],
    query: Annotated[str | None, Field(description="Optional keywords to scope the counts, "
                                                   "e.g. 'Kenya' for values among Kenya documents.")] = None,
    limit: Annotated[int, Field(ge=1, le=100, description="How many values to return (1-100).")] = 25,
) -> FacetResult:
    """DISCOVER: list the distinct values of a field with document counts, most frequent first.

    Use for questions like "what document types are published most?", "which languages exist?",
    "what topics are covered?", or to see the exact values before calling filter_documents.
    Fields: country, region, topic, sector, document_type, major_document_type, language.
    Returns {field, values: [{name, count}]}.
    """
    return await _api(api.facets(field, query, limit))


@mcp.tool
async def resolve_filter_values(
    field: Annotated[FacetField, Field(description="Field the terms belong to.")],
    terms: Annotated[list[str], BeforeValidator(_as_list),
                     Field(min_length=1, max_length=5, description="User wording to match, e.g. ['kenya'].")],
    max_matches: Annotated[int, Field(ge=1, le=10, description="Candidates per term.")] = 5,
) -> list[ResolveResult]:
    """RESOLVE: map the user's wording to exact catalogue values for filter_documents.

    Filters are exact-match, so call this first when unsure of spelling or naming, e.g.
    country 'kenya' -> 'Kenya', sector 'water and sanitation' ->
    'Water, Sanitation and Waste Management', region 'Sub-Saharan Africa' -> its regions.
    Returns one entry per term with candidate {value, count, score}; score 1.0 is exact.
    """
    return await _api(api.resolve(field, terms, max_matches))


@mcp.tool
async def filter_documents(
    country: FilterValues = None,
    region: FilterValues = None,
    topic: FilterValues = None,
    sector: FilterValues = None,
    document_type: FilterValues = None,
    major_document_type: FilterValues = None,
    language: FilterValues = None,
    start_date: Annotated[IsoDate | None, Field(description="Earliest document date, YYYY-MM-DD.")] = None,
    end_date: Annotated[IsoDate | None, Field(description="Latest document date, YYYY-MM-DD.")] = None,
    query: Annotated[str | None, Field(description="Optional extra keywords (full-text).")] = None,
    newest_first: Annotated[bool | None, Field(
        description="Sort newest first. Default: newest first without `query`, "
                    "by relevance when `query` is given.")] = None,
    rows: Rows = 5,
    offset: Offset = 0,
) -> SearchResult:
    """RETRIEVE: find documents by exact filters, optionally narrowed by keywords and dates.

    Use when the question names a country, region, topic, sector, document type
    (e.g. 'Project Documents', 'Economic & Sector Work' as major_document_type) or a date range.
    Values are matched case-insensitively; a value that does not exist returns an error with
    suggestions. Use resolve_filter_values or get_facets first if unsure.
    Returns {total, offset, rows, next_offset, documents: [{id, title, date, doc_type, country,
    language, abstract, pdf_url, url}], hint}. Page with offset=next_offset.
    """
    filters = {"country": country, "region": region, "topic": topic, "sector": sector,
               "document_type": document_type, "major_document_type": major_document_type,
               "language": language}
    if not any(filters.values()) and not (start_date or end_date):
        raise ToolError("Provide at least one filter (country, region, topic, sector, document_type, "
                        "major_document_type, language, start_date or end_date). "
                        "For keyword-only questions use search_documents.")
    if start_date and end_date and start_date > end_date:
        raise ToolError(f"start_date {start_date} is after end_date {end_date}.")

    params: dict[str, Any] = {"qterm": query, "strdate": start_date, "enddate": end_date}
    applied: dict[str, str] = {}
    for field, values in filters.items():
        if values:
            exact: list[str] = []
            for value in values:
                exact += await _api(api.normalize(field, value))
            applied[field] = params[FIELD_MAP[field]] = "^".join(dict.fromkeys(exact))
    if newest_first or (newest_first is None and not query):
        params.update(srt="docdt", order="desc")

    result = await _api(api.search(params, offset=offset, rows=rows))
    result.applied_filters = applied or None
    if result.total == 0:
        result.hint = ("No documents matched. Relax a filter, widen the dates, drop `query`, "
                       "or check values with get_facets.")
    return result


@mcp.tool
async def search_documents(
    query: Annotated[str, Field(min_length=2, description="Keywords, e.g. 'climate resilience Kenya'.")],
    language: Annotated[str | None, Field(description="Exact language, e.g. 'English', 'French'.")] = None,
    rows: Rows = 5,
    offset: Offset = 0,
) -> SearchResult:
    """RETRIEVE: full-text search across titles, abstracts, countries and project names.

    Use for keyword or subject questions with no structured filter (results ranked by relevance).
    Prefer filter_documents when the question names a country, document type or date range.
    Returns {total, offset, rows, next_offset, documents: [...], hint}. Page with offset=next_offset.
    """
    result = await _api(api.search({"qterm": query, "lang_exact": language}, offset=offset, rows=rows))
    if result.total == 0:
        result.hint = "No documents matched. Try fewer or broader keywords, or use get_facets to explore."
    return result


@mcp.tool
async def get_document(
    document_id: Annotated[str, Field(pattern=r"^D?\d+$",
                                      description="Document id from a search result, e.g. '33295056' or 'D33295056'.")],
) -> dict[str, Any]:
    """INSPECT: full metadata for one document (title, abstract, dates, country, project,
    authors, topics, PDF/text links, ...). Use after a search when the user wants details
    about a specific document. Returns the World Bank record as-is.
    """
    return await _api(api.get_document(document_id))


@mcp.prompt
def research_workflow(question: str) -> str:
    """How to answer a question about World Bank documents with these tools."""
    return f"{WORKFLOW}\n\nQuestion: {question}"


def main() -> None:
    mcp.run(show_banner=False)  # stdio transport


if __name__ == "__main__":
    main()
