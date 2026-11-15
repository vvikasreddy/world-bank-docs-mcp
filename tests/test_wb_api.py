"""Unit tests for the World Bank API client: parsing, pagination, error mapping."""

from __future__ import annotations

import re

import httpx
import pytest

from server.wb_api import (
    ABSTRACT_MAX_CHARS,
    WorldBankAPIError,
    next_offset,
    parse_documents,
    parse_facets,
)
from tests.conftest import RecordingRouter, client_with, load_fixture


def test_parse_documents_skips_facets_key():
    payload = {"documents": {"D1": {"id": "1", "display_title": "A"}, "facets": {"x": {}}}}
    docs = parse_documents(payload)
    assert [d.id for d in docs] == ["1"]


def test_parse_documents_maps_fields():
    doc = parse_documents(load_fixture("search_climate_kenya"))[0]
    assert doc.country == "Kenya"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc.date)
    assert doc.pdf_url.startswith("http")
    assert doc.title


def test_abstract_truncated():
    payload = {"documents": {"D1": {"id": "1", "display_title": "A",
                                    "abstracts": {"cdata!": "word " * 400}}}}
    abstract = parse_documents(payload)[0].abstract
    assert len(abstract) <= ABSTRACT_MAX_CHARS + 1
    assert abstract.endswith("…")


def test_abstract_accepts_plain_string_and_missing():
    payload = {"documents": {"D1": {"id": "1", "display_title": "A", "abstracts": "short"},
                             "D2": {"id": "2", "display_title": "B"}}}
    docs = parse_documents(payload)
    assert docs[0].abstract == "short"
    assert docs[1].abstract is None


def test_next_offset():
    assert next_offset(total=50, offset=0, rows=5) == 5
    assert next_offset(total=7, offset=5, rows=5) is None
    assert next_offset(total=0, offset=0, rows=5) is None


def test_parse_facets_sorted_by_count():
    values = parse_facets(load_fixture("facets_docty"), "docty_exact")
    counts = [v.count for v in values]
    assert counts == sorted(counts, reverse=True)
    assert values[0].name == "Procurement Plan"


async def test_search_empty_result():
    api = client_with(RecordingRouter())
    result = await api.search({"qterm": "zzqqxxnonexistent"}, offset=0, rows=5)
    assert result.total == 0
    assert result.documents == []
    assert result.next_offset is None


async def test_search_paginates():
    router = RecordingRouter()
    api = client_with(router)
    result = await api.search({"qterm": "climate Kenya"}, offset=10, rows=3)
    assert router.calls[-1]["os"] == "10"
    assert router.calls[-1]["rows"] == "3"
    assert result.offset == 10 and result.rows == 3
    assert result.total > 0


async def test_fetch_sends_format_json_and_drops_none():
    router = RecordingRouter()
    api = client_with(router)
    await api.fetch({"qterm": "x", "lang_exact": None})
    assert router.calls[-1]["format"] == "json"
    assert "lang_exact" not in router.calls[-1]


async def test_fetch_timeout_raises_api_error():
    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(WorldBankAPIError, match="timed out"):
        await client_with(boom).fetch({"qterm": "x"})


async def test_fetch_http_error_raises_api_error():
    with pytest.raises(WorldBankAPIError, match="503"):
        await client_with(lambda r: httpx.Response(503, text="down")).fetch({"qterm": "x"})


async def test_fetch_non_json_raises_api_error():
    with pytest.raises(WorldBankAPIError, match="JSON"):
        await client_with(lambda r: httpx.Response(200, text="<html>")).fetch({"qterm": "x"})


async def test_fetch_connection_error_raises_api_error():
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(WorldBankAPIError, match="reach"):
        await client_with(boom).fetch({"qterm": "x"})


async def test_get_document_strips_prefix_and_returns_metadata():
    router = RecordingRouter()
    doc = await client_with(router).get_document("D33295056")
    assert router.calls[-1]["id"] == "33295056"
    assert doc["display_title"]
    assert "facets" not in doc


async def test_get_document_not_found():
    with pytest.raises(WorldBankAPIError, match="No document found"):
        await client_with(RecordingRouter()).get_document("123")
