"""Contract tests against the real World Bank API. Run with:  pytest -m live

They guard against the captured fixtures drifting from the live service.
"""

from __future__ import annotations

import pytest

from server.wb_api import WorldBankAPIError, WorldBankClient

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
    assert doc.title and doc.url
    assert len(doc.model_dump_json()) < 4000


async def test_live_newest_first_sort(api):
    result = await api.search({"count_exact": "Indonesia", "srt": "docdt", "order": "desc"},
                              offset=0, rows=5)
    dates = [d.date for d in result.documents]
    assert dates == sorted(dates, reverse=True)


async def test_live_resolver_maps_real_values(api):
    assert await api.normalize("country", "kenya") == ["Kenya"]
    assert await api.normalize("sector", "water and sanitation") == [
        "Water, Sanitation and Waste Management"]


async def test_live_rare_spellings_include_official_name(api):
    assert "Egypt, Arab Republic of" in await api.normalize("country", "Egypt")
    assert await api.normalize("country", "vietnam") == ["Viet Nam", "Vietnam"]


async def test_live_ambiguous_country(api):
    with pytest.raises(WorldBankAPIError, match="ambiguous"):
        await api.normalize("country", "Congo")


async def test_live_all_languages_fit_one_call(api):
    from client.agent import MAX_TOOL_TEXT

    result = await api.facets("language", limit=200)
    assert not result.truncated and result.total_distinct > 100
    assert len(result.model_dump_json()) < MAX_TOOL_TEXT


async def test_live_full_page_fits_agent_budget(api):
    from client.agent import MAX_TOOL_TEXT

    page = await api.search({"qterm": "climate resilience Kenya"}, offset=0, rows=10)
    assert len(page.model_dump_json()) < MAX_TOOL_TEXT


async def test_answer_end_to_end_stdio():
    """Real MCP server subprocess over stdio + scripted LLM: the tool result reaches the LLM."""
    from client.agent import answer, load_settings
    from tests.test_agent import FakeLLM, reply, tool_call

    llm = FakeLLM([reply(tool_calls=[tool_call("c1", "get_facets", {"field": "language", "limit": 5})]),
                   reply("English is the most common language.")])
    result = await answer("Which languages?", load_settings({}), llm=llm)
    assert result == "English is the most common language."
    tool_message = next(m for m in llm.requests[1] if m["role"] == "tool")
    assert "English" in tool_message["content"]
