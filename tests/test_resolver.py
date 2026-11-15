"""Tests for mapping a user's wording to exact World Bank filter values."""

from __future__ import annotations

import pytest

from server.wb_api import FacetValue, WorldBankAPIError, rank_candidates
from tests.conftest import (
    COUNTRIES,
    COUNTRY_VARIANTS,
    REGIONS,
    SECTORS,
    RecordingRouter,
    client_with,
    facet_router,
)


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
    tied = [FacetValue(name="Water Supply", count=10), FacetValue(name="Water Policy", count=100)]
    assert [m.value for m in rank_candidates("water", tied, 2)] == ["Water Policy", "Water Supply"]


def test_resolve_prefers_closer_value():
    names = [m.value for m in rank_candidates("water", SECTORS, 5)]
    assert names.index("Water Resources") < names.index("Water, Sanitation and Waste Management")


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
    api = client_with(facet_router({"admreg_exact": REGIONS}))
    values = await api.normalize("region", "Sub-Saharan Africa")
    assert "Eastern and Southern Africa" in values and "Western and Central Africa" in values


async def test_alias_matches_report_real_counts():
    api = client_with(facet_router({"admreg_exact": REGIONS}))
    [result] = await api.resolve("region", ["Sub-Saharan Africa"], 5)
    assert all(m.count > 0 for m in result.matches)


def variants_api():
    return client_with(facet_router({"count_exact": COUNTRY_VARIANTS}))


async def test_rare_exact_spelling_combined_with_dominant_variant():
    # "Egypt" exists with 14 docs; the official "Egypt, Arab Republic of" has 4,381.
    assert await variants_api().normalize("country", "Egypt") == ["Egypt, Arab Republic of", "Egypt"]


async def test_small_variants_not_combined():
    assert await variants_api().normalize("country", "Macedonia") == ["North Macedonia", "Macedonia"]


async def test_several_dominant_variants_is_ambiguous():
    with pytest.raises(WorldBankAPIError, match=r"ambiguous.*Congo, Democratic Republic of"):
        await variants_api().normalize("country", "Congo")


async def test_country_alias_uses_catalogue_spelling():
    assert await variants_api().normalize("country", "vietnam") == ["Viet Nam", "Vietnam"]


async def test_empty_facet_list_is_an_error_and_not_cached():
    router = facet_router({"count_exact": []})
    api = client_with(router)
    for _ in range(2):
        with pytest.raises(WorldBankAPIError, match="unavailable"):
            await api.normalize("country", "Kenya")
    assert sum(1 for c in router.calls if c.get("fct") == "count_exact") == 2


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


async def test_partial_match_dropping_user_words_is_not_auto_applied():
    # "education financing" -> "Education" would silently drop "financing"
    api = client_with(facet_router({"sectr_exact": SECTORS}))
    with pytest.raises(WorldBankAPIError, match=r"Education.*query"):
        await api.normalize("sector", "education financing")


def test_partial_match_still_suggested():
    assert rank_candidates("education financing", SECTORS, 3)[0].value == "Education"


DOC_TYPES = [FacetValue(name="Project Document", count=151),
             FacetValue(name="Project Preparation Facility Documents", count=1),
             FacetValue(name="GEF Project Document", count=110)]


def test_plural_and_closest_value_preferred():
    assert rank_candidates("Project Documents", DOC_TYPES, 3)[0].value == "Project Document"


async def test_normalize_prefers_closest_subset_match():
    api = client_with(facet_router({"docty_exact": DOC_TYPES}))
    assert await api.normalize("document_type", "project documents") == ["Project Document"]


async def test_legacy_fy_prefixed_values_ignored_when_resolving():
    sectors = SECTORS + [FacetValue(name="FY17 - Water, Sanitation and Waste Management", count=99999)]
    api = client_with(facet_router({"sectr_exact": sectors}))
    assert await api.normalize("sector", "water and sanitation") == [
        "Water, Sanitation and Waste Management"]


async def test_normalize_unknown_suggests():
    api = client_with(countries_router())
    with pytest.raises(WorldBankAPIError, match="Did you mean.*Kenya"):
        await api.normalize("country", "Kenia")
