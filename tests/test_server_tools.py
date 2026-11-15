"""Tests for the MCP tools, exercised through an in-memory FastMCP client (no network)."""

from __future__ import annotations

import httpx
import pytest
from fastmcp import Client

from server import wb_docs_mcp
from tests.conftest import COUNTRIES, SECTORS, RecordingRouter, client_with, facet_router

REQUIRED_TOOLS = {"search_documents", "filter_documents", "get_document", "get_facets"}


@pytest.fixture
def router(monkeypatch) -> RecordingRouter:
    router = facet_router({"count_exact": COUNTRIES, "sectr_exact": SECTORS})
    monkeypatch.setattr(wb_docs_mcp, "api", client_with(router))
    return router


async def call(name: str, args: dict):
    async with Client(wb_docs_mcp.mcp) as client:
        return await client.call_tool(name, args, raise_on_error=False)


def text_of(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


async def test_lists_required_and_extra_tools():
    async with Client(wb_docs_mcp.mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert REQUIRED_TOOLS <= set(tools)
    assert "resolve_filter_values" in tools
    assert all(len(t.description or "") > 40 for t in tools.values())


async def test_search_maps_params(router):
    result = await call("search_documents",
                        {"query": "climate Kenya", "language": "English", "rows": 3, "offset": 6})
    assert not result.is_error, text_of(result)
    sent = router.last_search()
    assert sent["qterm"] == "climate Kenya"
    assert sent["lang_exact"] == "English"
    assert sent["rows"] == "3" and sent["os"] == "6"
    assert result.structured_content["documents"]


async def test_search_rejects_rows_over_20(router):
    result = await call("search_documents", {"query": "climate", "rows": 50})
    assert result.is_error


async def test_filter_requires_at_least_one_filter(router):
    result = await call("filter_documents", {})
    assert result.is_error and "at least one" in text_of(result)


async def test_filter_rejects_bad_date(router):
    result = await call("filter_documents", {"country": "Kenya", "start_date": "2019"})
    assert result.is_error


async def test_filter_rejects_start_after_end(router):
    result = await call("filter_documents",
                        {"country": "Kenya", "start_date": "2022-01-01", "end_date": "2019-01-01"})
    assert result.is_error and "start_date" in text_of(result)


async def test_filter_normalizes_values_and_maps_params(router):
    result = await call("filter_documents", {
        "country": "kenya", "sector": "water and sanitation",
        "start_date": "2019-01-01", "end_date": "2022-12-31"})
    assert not result.is_error, text_of(result)
    sent = router.last_search()
    assert sent["count_exact"] == "Kenya"
    assert sent["sectr_exact"] == "Water, Sanitation and Waste Management"
    assert sent["strdate"] == "2019-01-01" and sent["enddate"] == "2022-12-31"
    assert result.structured_content["applied_filters"]["sector"] == "Water, Sanitation and Waste Management"


async def test_filter_ambiguous_value_errors_with_options(router):
    result = await call("filter_documents", {"sector": "water"})
    assert result.is_error
    assert "Water Resources" in text_of(result)


async def test_filter_unknown_value_suggests(router):
    result = await call("filter_documents", {"country": "Kenia"})
    assert result.is_error and "Kenya" in text_of(result)


async def test_filter_accepts_list_and_joins_with_or(router):
    result = await call("filter_documents", {"country": ["Kenya", "Brazil"]})
    assert not result.is_error, text_of(result)
    assert router.last_search()["count_exact"] == "Kenya^Brazil"


async def test_filter_region_alias(router):
    result = await call("filter_documents", {"region": "Sub-Saharan Africa"})
    assert not result.is_error, text_of(result)
    assert "Eastern and Southern Africa^Western and Central Africa" in router.last_search()["admreg_exact"]


async def test_filter_without_query_sorts_newest_first(router):
    await call("filter_documents", {"country": "Indonesia"})
    sent = router.last_search()
    assert sent["srt"] == "docdt" and sent["order"] == "desc"


async def test_filter_with_query_keeps_relevance_order_by_default(router):
    await call("filter_documents", {"country": "Kenya", "query": "education financing"})
    assert "srt" not in router.last_search()


async def test_filter_newest_first_explicit_overrides(router):
    await call("filter_documents", {"country": "Kenya", "query": "climate", "newest_first": True})
    assert router.last_search()["srt"] == "docdt"


async def test_filter_zero_results_includes_hint(router):
    result = await call("filter_documents", {"country": "Kenya", "query": "zzqqxxnonexistent"})
    assert not result.is_error
    assert "get_facets" in (result.structured_content.get("hint") or "")


async def test_get_document_accepts_D_prefix(router):
    result = await call("get_document", {"document_id": "D33295056"})
    assert not result.is_error, text_of(result)
    assert router.calls[-1]["id"] == "33295056"
    assert result.structured_content["display_title"]


async def test_get_document_rejects_non_numeric_id(router):
    result = await call("get_document", {"document_id": "abc"})
    assert result.is_error


async def test_get_facets_returns_sorted_values(router):
    result = await call("get_facets", {"field": "document_type", "limit": 3})
    assert not result.is_error, text_of(result)
    values = result.structured_content["values"]
    assert len(values) == 3 and values[0]["name"] == "Procurement Plan"


async def test_get_facets_rejects_unknown_field(router):
    result = await call("get_facets", {"field": "colour"})
    assert result.is_error


async def test_resolve_filter_values_tool(router):
    result = await call("resolve_filter_values", {"field": "country", "terms": ["kenya", "brazil"]})
    assert not result.is_error, text_of(result)
    results = result.structured_content["result"]
    assert [r["matches"][0]["value"] for r in results] == ["Kenya", "Brazil"]


async def test_api_failure_becomes_tool_error(monkeypatch):
    monkeypatch.setattr(wb_docs_mcp, "api", client_with(lambda r: httpx.Response(500)))
    result = await call("search_documents", {"query": "climate"})
    assert result.is_error and "World Bank API" in text_of(result)
