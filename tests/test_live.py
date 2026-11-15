"""Contract tests against the real World Bank API. Run with:  pytest -m live

They guard against the captured fixtures drifting from the live service.
"""

from __future__ import annotations

import pytest

from server.wb_api import WorldBankClient

pytestmark = pytest.mark.live


@pytest.fixture
def api() -> WorldBankClient:
    return WorldBankClient()


async def test_live_search_returns_docs(api):
    result = await api.search({"qterm": "climate resilience Kenya"}, offset=0, rows=3)
    assert result.total > 0
    assert result.documents and result.documents[0].title


async def test_live_language_facets_include_english(api):
    result = await api.facets("language", limit=10)
    assert "English" in [v.name for v in result.values]


async def test_live_get_document_by_search_id(api):
    doc_id = (await api.search({"qterm": "education"}, offset=0, rows=1)).documents[0].id
    doc = await api.get_document(doc_id)
    assert doc["display_title"]


async def test_live_newest_first_sort(api):
    result = await api.search({"count_exact": "Indonesia", "srt": "docdt", "order": "desc"},
                              offset=0, rows=5)
    dates = [d.date for d in result.documents]
    assert dates == sorted(dates, reverse=True)


async def test_live_resolver_maps_real_values(api):
    assert await api.normalize("country", "kenya") == ["Kenya"]
    assert await api.normalize("sector", "water and sanitation") == [
        "Water, Sanitation and Waste Management"]
