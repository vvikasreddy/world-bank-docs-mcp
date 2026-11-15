"""Tests for mapping a user's wording to exact World Bank filter values."""

from __future__ import annotations

import pytest

from server.wb_api import FacetValue, WorldBankAPIError, rank_candidates
from tests.conftest import COUNTRIES, SECTORS, RecordingRouter, client_with, facet_router


def countries_router() -> RecordingRouter:
    return facet_router({"count_exact": COUNTRIES})


def test_resolve_case_insensitive_exact():
    best = rank_candidates("kenya", COUNTRIES, 3)[0]
    assert best.value == "Kenya" and best.score == 1.0


def test_resolve_token_match_ignores_punctuation_and_order():
    best = rank_candidates("water and sanitation", SECTORS, 3)[0]
    assert best.value == "Water, Sanitation and Waste Management"


def test_resolve_substring():
    names = [m.value for m in rank_candidates("water", SECTORS, 5)]
    assert "Water Resources" in names and "Water, Sanitation and Waste Management" in names


def test_resolve_typo():
    assert rank_candidates("Kenia", COUNTRIES, 3)[0].value == "Kenya"


def test_resolve_ranks_ties_by_count():
    names = [m.value for m in rank_candidates("water", SECTORS, 5)]
    assert names.index("Water, Sanitation and Waste Management") < names.index("Water Resources")


def test_resolve_no_match_returns_empty():
    assert rank_candidates("zzzz", COUNTRIES, 3) == []


async def test_facet_values_cached():
    router = countries_router()
    api = client_with(router)
    await api.resolve("country", ["kenya"], 3)
    await api.resolve("country", ["brazil"], 3)
    assert sum(1 for c in router.calls if c.get("fct") == "count_exact") == 1


async def test_resolve_returns_one_result_per_term():
    api = client_with(countries_router())
    results = await api.resolve("country", ["kenya", "brazil"], 3)
    assert [r.term for r in results] == ["kenya", "brazil"]
    assert results[1].matches[0].value == "Brazil"


async def test_region_alias_sub_saharan_africa():
    api = client_with(RecordingRouter())
    values = await api.normalize("region", "Sub-Saharan Africa")
    assert "Eastern and Southern Africa" in values and "Western and Central Africa" in values


async def test_normalize_exact_case_insensitive():
    api = client_with(countries_router())
    assert await api.normalize("country", "  kenya ") == ["Kenya"]


async def test_normalize_accepts_confident_unique_match():
    api = client_with(facet_router({"sectr_exact": SECTORS}))
    assert await api.normalize("sector", "water and sanitation") == [
        "Water, Sanitation and Waste Management"]


async def test_normalize_ambiguous_raises():
    api = client_with(facet_router({"sectr_exact": SECTORS}))
    with pytest.raises(WorldBankAPIError, match="Water Resources"):
        await api.normalize("sector", "water")


async def test_legacy_fy_prefixed_values_ignored_when_resolving():
    sectors = SECTORS + [FacetValue(name="FY17 - Water, Sanitation and Waste Management", count=99999)]
    api = client_with(facet_router({"sectr_exact": sectors}))
    assert await api.normalize("sector", "water and sanitation") == [
        "Water, Sanitation and Waste Management"]


async def test_normalize_unknown_suggests():
    api = client_with(countries_router())
    with pytest.raises(WorldBankAPIError, match="Did you mean.*Kenya"):
        await api.normalize("country", "Kenia")
