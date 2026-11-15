"""Shared test helpers: real API fixtures served through httpx.MockTransport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from server.wb_api import FacetValue, WorldBankClient

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def client_with(handler: Callable[[httpx.Request], httpx.Response]) -> WorldBankClient:
    """A WorldBankClient whose HTTP layer is replaced by `handler`."""
    return WorldBankClient(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def route_to_fixture(request: httpx.Request) -> httpx.Response:
    """Pick a captured real response based on the query parameters."""
    params = request.url.params
    if "fct" in params:
        field = params["fct"]
        name = {"docty_exact": "facets_docty", "count_exact": "facets_country"}.get(field)
        if name is None:
            return httpx.Response(200, json={"rows": 0, "os": 0, "page": 1, "total": 0,
                                             "documents": {"facets": {field: {}}}})
        return httpx.Response(200, json=load_fixture(name))
    if "id" in params:
        if params["id"] == "33295056":
            return httpx.Response(200, json=load_fixture("get_document"))
        return httpx.Response(200, json=load_fixture("empty"))
    if params.get("qterm") == "zzqqxxnonexistent":
        return httpx.Response(200, json=load_fixture("empty"))
    return httpx.Response(200, json=load_fixture("search_climate_kenya"))


class RecordingRouter:
    """Serves fixtures and records every request's query params."""

    def __init__(self, override: Callable[[httpx.Request], httpx.Response | None] | None = None):
        self.calls: list[dict[str, str]] = []
        self._override = override

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(dict(request.url.params))
        if self._override is not None:
            response = self._override(request)
            if response is not None:
                return response
        return route_to_fixture(request)

    def last_search(self) -> dict[str, str]:
        """Most recent request that was a document query (not a facet lookup)."""
        return next(c for c in reversed(self.calls) if "fct" not in c)


def facet_payload(api_field: str, values: list[FacetValue]) -> dict:
    """A facet response in the real API's shape."""
    return {"rows": 0, "os": 0, "page": 1, "total": 0, "documents": {"facets": {api_field: {
        str(i): {"name": v.name, "label": v.name, "count": v.count} for i, v in enumerate(values)}}}}


def facet_router(facets: dict[str, list[FacetValue]]) -> RecordingRouter:
    """Router that answers facet requests for the given API fields from `facets`."""
    def override(request: httpx.Request):
        field = request.url.params.get("fct")
        if field in facets:
            return httpx.Response(200, json=facet_payload(field, facets[field]))
        return None
    return RecordingRouter(override)


COUNTRIES = [FacetValue(name=n, count=c) for n, c in
             [("Kenya", 9738), ("Brazil", 8000), ("Benin", 3000), ("Kyrgyz Republic", 2000),
              ("Indonesia", 7000), ("World", 50000)]]
SECTORS = [FacetValue(name=n, count=c) for n, c in
           [("Water, Sanitation and Waste Management", 40000), ("Education", 30000),
            ("Water Resources", 1000), ("Transportation", 20000)]]


@pytest.fixture
def router() -> RecordingRouter:
    return RecordingRouter()
