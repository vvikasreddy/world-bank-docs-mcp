"""Thin async client for the World Bank Documents & Reports API (search.worldbank.org/api/v3/wds).

This module is the only place that performs HTTP. It knows nothing about MCP: it returns
Pydantic models and raises `WorldBankAPIError` with messages written to be read by an LLM.

API behaviour verified against the live service (2026-11-14):
- Documents come back in `payload["documents"]`, a dict keyed "D<id>". When facets are
  requested the same dict also holds a "facets" key, which is not a document.
- `id=<number>` looks up a single document; an unknown id returns total=0.
- `srt=docdt&order=desc` sorts newest first.
- `*_exact` filters are case-sensitive ("kenya" -> 0 hits, "Kenya" -> 9,738) and accept
  several values joined with "^" (OR).
- `topic_exact` matches nothing; topics live in `teratopic_exact` (29 values).
- Facet lists default to 100 values; `facet.limit` raises that (there are 488 countries).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

API_BASE = "https://search.worldbank.org/api/v3/wds"
ABSTRACT_MAX_CHARS = 300
DEFAULT_TIMEOUT = 20.0

# Friendly field names exposed to the LLM -> the API's exact-match field names.
FacetField = Literal[
    "country", "region", "topic", "sector", "document_type", "major_document_type", "language"
]
FIELD_MAP: dict[str, str] = {
    "country": "count_exact",
    "region": "admreg_exact",
    "topic": "teratopic_exact",
    "sector": "sectr_exact",
    "document_type": "docty_exact",
    "major_document_type": "majdocty_exact",
    "language": "lang_exact",
}

# Fields returned for list views: small enough to keep LLM context (and Groq TPM) in check.
SUMMARY_FIELDS = "id,display_title,docdt,docty,majdocty,count,lang,abstracts,pdfurl,url"


class WorldBankAPIError(Exception):
    """The World Bank API failed or returned something unusable. Message is user/LLM-facing."""


class DocumentSummary(BaseModel):
    id: str
    title: str
    date: str | None = Field(None, description="Document date, YYYY-MM-DD")
    doc_type: str | None = None
    country: str | None = None
    language: str | None = None
    abstract: str | None = Field(None, description=f"First ~{ABSTRACT_MAX_CHARS} characters")
    pdf_url: str | None = None
    url: str | None = None


class SearchResult(BaseModel):
    total: int = Field(description="Total matching documents in the World Bank catalogue")
    offset: int
    rows: int
    next_offset: int | None = Field(description="Pass as `offset` to get the next page; null if none")
    documents: list[DocumentSummary]
    applied_filters: dict[str, str] | None = Field(
        None, description="Exact catalogue values the filters were resolved to")
    hint: str | None = None


class FacetValue(BaseModel):
    name: str
    count: int


class FacetResult(BaseModel):
    field: str
    values: list[FacetValue] = Field(description="Distinct values, most frequent first")


class Match(BaseModel):
    value: str = Field(description="Exact World Bank value to pass to filter_documents")
    count: int = Field(description="Number of documents with this value")
    score: float = Field(description="Match confidence, 1.0 = exact")


class ResolveResult(BaseModel):
    field: str
    term: str
    matches: list[Match]


# Common names the catalogue does not use verbatim.
REGION_ALIASES: dict[str, list[str]] = {
    "sub saharan africa": ["Eastern and Southern Africa", "Western and Central Africa", "Africa"],
    "ssa": ["Eastern and Southern Africa", "Western and Central Africa", "Africa"],
    "middle east and north africa": ["Middle East, North Africa, Afghanistan, and Pakistan"],
    "mena": ["Middle East, North Africa, Afghanistan, and Pakistan"],
}
FIELD_ALIASES: dict[str, dict[str, list[str]]] = {"region": REGION_ALIASES}

FACET_CACHE_LIMIT = 1000
# A non-exact match is applied automatically only if it is this good AND beats the runner-up.
CONFIDENT_SCORE = 0.85
# Legacy taxonomy duplicates such as "FY17 - Water, Sanitation and Waste Management".
_LEGACY_VALUE = re.compile(r"^FY\d{2} - ")
_STOPWORDS = {"and", "of", "the", "for", "in", "on", "to", "a", "&"}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def _tokens(text: str) -> set[str]:
    return {t for t in _norm(text).split() if t not in _STOPWORDS}


def _score(term: str, candidate: str) -> float:
    """0.0 (no match) or a confidence in [0.6, 1.0]."""
    t_norm, c_norm = _norm(term), _norm(candidate)
    if not t_norm:
        return 0.0
    if t_norm == c_norm:
        return 1.0
    t_tok, c_tok = _tokens(term), _tokens(candidate)
    if t_tok and c_tok and (t_tok <= c_tok or c_tok <= t_tok):
        return 0.85
    coverage = len(t_tok & c_tok) / len(t_tok) if t_tok else 0.0
    fuzzy = SequenceMatcher(None, t_norm, c_norm).ratio()
    return max(0.5 + 0.3 * coverage if coverage >= 0.5 else 0.0,
               fuzzy * 0.8 if fuzzy >= 0.75 else 0.0)


def rank_candidates(term: str, candidates: list[FacetValue], max_matches: int) -> list[Match]:
    """Best facet values for `term`: exact > all-words > partial words > fuzzy; ties by doc count."""
    scored = [Match(value=c.name, count=c.count, score=round(_score(term, c.name), 2))
              for c in candidates]
    scored = [m for m in scored if m.score > 0]
    scored.sort(key=lambda m: (m.score, m.count), reverse=True)
    return scored[:max_matches]


def next_offset(total: int, offset: int, rows: int) -> int | None:
    nxt = offset + rows
    return nxt if nxt < total else None


def _abstract(raw: Any) -> str | None:
    text = raw.get("cdata!") if isinstance(raw, dict) else raw
    if not isinstance(text, str) or not text.strip():
        return None
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= ABSTRACT_MAX_CHARS:
        return text
    return text[:ABSTRACT_MAX_CHARS].rsplit(" ", 1)[0] + "…"


def _iter_documents(payload: dict) -> list[dict]:
    docs = payload.get("documents") or {}
    return [doc for key, doc in docs.items() if key != "facets" and isinstance(doc, dict)]


def parse_documents(payload: dict) -> list[DocumentSummary]:
    return [
        DocumentSummary(
            id=str(doc.get("id", "")),
            title=doc.get("display_title") or "(untitled)",
            date=(doc.get("docdt") or "")[:10] or None,
            doc_type=doc.get("docty"),
            country=doc.get("count"),
            language=doc.get("lang"),
            abstract=_abstract(doc.get("abstracts")),
            pdf_url=doc.get("pdfurl"),
            url=doc.get("url"),
        )
        for doc in _iter_documents(payload)
    ]


def parse_facets(payload: dict, api_field: str) -> list[FacetValue]:
    raw = ((payload.get("documents") or {}).get("facets") or {}).get(api_field) or {}
    values = [FacetValue(name=v["name"], count=int(v["count"]))
              for v in raw.values() if isinstance(v, dict) and v.get("name")]
    return sorted(values, key=lambda v: v.count, reverse=True)


class WorldBankClient:
    def __init__(self, http: httpx.AsyncClient | None = None, timeout: float = DEFAULT_TIMEOUT,
                 base_url: str = API_BASE):
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._base_url = base_url
        self._timeout = timeout
        self._facet_cache: dict[str, list[FacetValue]] = {}

    async def fetch(self, params: dict[str, Any]) -> dict:
        """Single HTTP call site. Drops None params, always asks for JSON."""
        query = {k: v for k, v in params.items() if v is not None}
        query["format"] = "json"
        try:
            response = await self._http.get(self._base_url, params=query)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise WorldBankAPIError(
                f"The World Bank API timed out after {self._timeout:.0f}s. Try again or narrow the query."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise WorldBankAPIError(
                f"The World Bank API returned HTTP {exc.response.status_code}. Try again shortly."
            ) from exc
        except httpx.HTTPError as exc:
            raise WorldBankAPIError(f"Could not reach the World Bank API: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise WorldBankAPIError("The World Bank API returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise WorldBankAPIError("The World Bank API returned an unexpected JSON shape.")
        return data

    async def search(self, params: dict[str, Any], offset: int, rows: int) -> SearchResult:
        payload = await self.fetch({**params, "fl": SUMMARY_FIELDS, "rows": rows, "os": offset})
        total = int(payload.get("total") or 0)
        return SearchResult(
            total=total,
            offset=offset,
            rows=rows,
            next_offset=next_offset(total, offset, rows),
            documents=parse_documents(payload),
        )

    async def get_document(self, doc_id: str) -> dict:
        numeric = doc_id.strip().removeprefix("D")
        payload = await self.fetch({"id": numeric})
        docs = _iter_documents(payload)
        if not docs:
            raise WorldBankAPIError(f"No document found with id '{doc_id}'.")
        doc = dict(docs[0])
        if not doc.get("facets"):
            doc.pop("facets", None)
        return doc

    async def facets(self, field: str, query: str | None = None, limit: int = 25) -> FacetResult:
        api_field = FIELD_MAP[field]
        payload = await self.fetch({"fct": api_field, "rows": 0, "qterm": query,
                                    "facet.limit": max(limit, 100)})
        return FacetResult(field=field, values=parse_facets(payload, api_field)[:limit])

    async def facet_values(self, field: str) -> list[FacetValue]:
        """Every distinct value of a field across the whole catalogue (cached per process)."""
        if field not in self._facet_cache:
            api_field = FIELD_MAP[field]
            payload = await self.fetch({"fct": api_field, "rows": 0,
                                        "facet.limit": FACET_CACHE_LIMIT})
            self._facet_cache[field] = [v for v in parse_facets(payload, api_field)
                                        if not _LEGACY_VALUE.match(v.name)]
        return self._facet_cache[field]

    async def resolve(self, field: str, terms: list[str], max_matches: int = 5) -> list[ResolveResult]:
        results = []
        for term in terms:
            alias = FIELD_ALIASES.get(field, {}).get(_norm(term))
            if alias:
                matches = [Match(value=v, count=0, score=1.0) for v in alias]
            else:
                matches = rank_candidates(term, await self.facet_values(field), max_matches)
            results.append(ResolveResult(field=field, term=term, matches=matches))
        return results

    async def normalize(self, field: str, value: str) -> list[str]:
        """Exact catalogue value(s) for `value` (case/spacing-insensitive, aliases expanded).

        Raises WorldBankAPIError with suggestions when nothing matches exactly, so the caller
        can self-correct instead of silently getting zero results.
        """
        alias = FIELD_ALIASES.get(field, {}).get(_norm(value))
        if alias:
            return list(alias)
        candidates = await self.facet_values(field)
        for candidate in candidates:
            if _norm(candidate.name) == _norm(value):
                return [candidate.name]
        ranked = rank_candidates(value, candidates, 5)
        if ranked and ranked[0].score >= CONFIDENT_SCORE and (
                len(ranked) == 1 or ranked[0].score > ranked[1].score):
            return [ranked[0].value]
        suggestions = [m.value for m in ranked]
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else (
            f" Call resolve_filter_values or get_facets with field='{field}' to see valid values.")
        raise WorldBankAPIError(f"No {field.replace('_', ' ')} named '{value.strip()}'.{hint}")
