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
ABSTRACT_MAX_CHARS = 200  # keeps a 10-row page under the agent's per-tool-result budget
DETAIL_ABSTRACT_MAX_CHARS = 1500
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


class DocumentDetail(BaseModel):
    """The useful subset of a full World Bank record (the raw record is ~10k+ characters)."""
    id: str
    title: str
    date: str | None = Field(None, description="Document date, YYYY-MM-DD")
    doc_type: str | None = None
    major_doc_type: str | None = None
    country: str | None = None
    region: str | None = None
    language: str | None = None
    project_id: str | None = None
    project_name: str | None = None
    report_number: str | None = None
    authors: list[str] = Field(default_factory=list)
    topics: str | None = None
    abstract: str | None = Field(None, description=f"First ~{DETAIL_ABSTRACT_MAX_CHARS} characters")
    pdf_url: str | None = None
    txt_url: str | None = None
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
    total_distinct: int = Field(description="How many distinct values exist")
    truncated: bool = Field(description="True if more values exist; raise `limit` to see them")
    hint: str | None = None


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
COUNTRY_ALIASES: dict[str, list[str]] = {
    "vietnam": ["Viet Nam", "Vietnam"],
    "turkey": ["Turkiye", "Turkey"],
    "south korea": ["Korea, Republic of"],
    "north korea": ["Korea, Democratic People's Republic of"],
    "drc": ["Congo, Democratic Republic of"],
    "democratic republic of congo": ["Congo, Democratic Republic of"],
    "ivory coast": ["Cote d'Ivoire"],
    "usa": ["United States"],
    "us": ["United States"],
    "uk": ["United Kingdom"],
}
FIELD_ALIASES: dict[str, dict[str, list[str]]] = {"region": REGION_ALIASES,
                                                  "country": COUNTRY_ALIASES}

FACET_CACHE_LIMIT = 1000
# An exact match is widened to include a variant spelling that contains all the user's words
# when that variant holds at least DOMINANT_RATIO times as many documents (e.g. "Egypt" (14) ->
# also "Egypt, Arab Republic of" (4,381)). Variants below DOMINANT_SHARE of the biggest one are
# ignored; two or more dominant variants ("Congo") make the value ambiguous.
DOMINANT_RATIO = 2.0
DOMINANT_SHARE = 0.25
# A non-exact match is applied automatically only if it is this good AND beats the
# runner-up by at least CONFIDENT_MARGIN (otherwise the caller gets suggestions instead).
CONFIDENT_SCORE = 0.85
CONFIDENT_MARGIN = 0.05
# Legacy taxonomy duplicates such as "FY17 - Water, Sanitation and Waste Management".
_LEGACY_VALUE = re.compile(r"^FY\d{2} - ")
_STOPWORDS = {"and", "of", "the", "for", "in", "on", "to", "a", "&"}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def _singular(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def _tokens(text: str) -> set[str]:
    return {_singular(t) for t in _norm(text).split() if t not in _STOPWORDS}


def _score(term: str, candidate: str) -> float:
    """0.0 (no match) or a confidence in [0.6, 1.0]."""
    t_norm, c_norm = _norm(term), _norm(candidate)
    if not t_norm:
        return 0.0
    if t_norm == c_norm:
        return 1.0
    t_tok, c_tok = _tokens(term), _tokens(candidate)
    if t_tok and t_tok == c_tok:   # same words up to plurals/stopwords/order
        return 0.98
    if t_tok and t_tok <= c_tok:   # every user word is in the value; fewer extra words = closer
        return 0.85 + 0.1 * len(t_tok) / len(c_tok)
    if c_tok and c_tok <= t_tok:   # value drops some user words ("education financing" -> "Education")
        return 0.7
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


def _abstract(raw: Any, max_chars: int = ABSTRACT_MAX_CHARS) -> str | None:
    text = raw.get("cdata!") if isinstance(raw, dict) else raw
    if not isinstance(text, str) or not text.strip():
        return None
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _iter_documents(payload: dict) -> list[dict]:
    docs = payload.get("documents") or {}
    return [doc for key, doc in docs.items() if key != "facets" and isinstance(doc, dict)]


def _flatten(raw: Any, key: str) -> list[str]:
    """{"0": {"author": "A"}, "1": {"author": "B"}} -> ["A", "B"] (the API's list encoding)."""
    if isinstance(raw, dict):
        items = [v.get(key) if isinstance(v, dict) else v for v in raw.values()]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def _date(doc: dict) -> str | None:
    return (doc.get("docdt") or "")[:10] or None


def _guard_parse(build):
    """Turn a malformed API record into a readable WorldBankAPIError instead of a crash."""
    def wrapper(*args, **kwargs):
        try:
            return build(*args, **kwargs)
        except (KeyError, TypeError, ValueError) as exc:  # pydantic ValidationError is a ValueError
            raise WorldBankAPIError(f"The World Bank API returned an unexpected record: {exc}") from exc
    return wrapper


@_guard_parse
def parse_documents(payload: dict) -> list[DocumentSummary]:
    return [
        DocumentSummary(
            id=str(doc.get("id", "")),
            title=doc.get("display_title") or "(untitled)",
            date=_date(doc),
            doc_type=doc.get("docty"),
            country=doc.get("count"),
            language=doc.get("lang"),
            abstract=_abstract(doc.get("abstracts")),
            pdf_url=doc.get("pdfurl"),
            url=doc.get("url"),
        )
        for doc in _iter_documents(payload)
    ]


@_guard_parse
def parse_document_detail(doc: dict) -> DocumentDetail:
    abstract = _abstract(doc.get("abstracts"), DETAIL_ABSTRACT_MAX_CHARS)
    return DocumentDetail(
        id=str(doc.get("id", "")),
        title=doc.get("display_title") or "(untitled)",
        date=_date(doc),
        doc_type=doc.get("docty"),
        major_doc_type=doc.get("majdocty"),
        country=doc.get("count"),
        region=doc.get("admreg"),
        language=doc.get("lang"),
        project_id=doc.get("projectid"),
        project_name=doc.get("projn"),
        report_number=doc.get("repnb"),
        authors=_flatten(doc.get("authors"), "author"),
        topics=doc.get("teratopic"),
        abstract=abstract,
        pdf_url=doc.get("pdfurl"),
        txt_url=doc.get("txturl"),
        url=doc.get("url"),
    )


@_guard_parse
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

    async def _get(self, query: dict[str, Any]) -> httpx.Response:
        """GET with one retry for transient failures (timeouts, 5xx). 4xx is never retried."""
        for attempt in range(2):
            try:
                response = await self._http.get(self._base_url, params=query)
                if response.status_code < 500 or attempt == 1:
                    return response.raise_for_status()
            except httpx.TimeoutException:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")

    async def fetch(self, params: dict[str, Any]) -> dict:
        """Single HTTP call site. Drops None params, always asks for JSON."""
        query = {k: v for k, v in params.items() if v is not None}
        query["format"] = "json"
        try:
            response = await self._get(query)
        except httpx.TimeoutException as exc:
            raise WorldBankAPIError(
                f"The World Bank API timed out after {self._timeout:.0f}s. Try again or narrow the query."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if 400 <= status < 500:
                raise WorldBankAPIError(
                    f"The World Bank API rejected the request (HTTP {status}). Check the dates, "
                    "offset and filter values; retrying the same request will not help."
                ) from exc
            raise WorldBankAPIError(
                f"The World Bank API returned HTTP {status}. Try again shortly."
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

    async def search(self, params: dict[str, Any], offset: int, rows: int,
                     empty_hint: str | None = None) -> SearchResult:
        payload = await self.fetch({**params, "fl": SUMMARY_FIELDS, "rows": rows, "os": offset})
        total = int(payload.get("total") or 0)
        hint = None
        if total == 0:
            hint = empty_hint
        elif offset >= total:
            hint = f"offset {offset} is past the last result (total {total}); use an offset below {total}."
        return SearchResult(
            total=total,
            offset=offset,
            rows=rows,
            next_offset=next_offset(total, offset, rows),
            documents=parse_documents(payload),
            hint=hint,
        )

    async def get_document(self, doc_id: str) -> DocumentDetail:
        numeric = doc_id.strip().removeprefix("D")
        docs = _iter_documents(await self.fetch({"id": numeric}))
        if not docs:
            raise WorldBankAPIError(f"No document found with id '{doc_id}'.")
        return parse_document_detail(docs[0])

    async def _all_facet_values(self, field: str, query: str | None = None) -> list[FacetValue]:
        api_field = FIELD_MAP[field]
        payload = await self.fetch({"fct": api_field, "rows": 0, "qterm": query,
                                    "facet.limit": FACET_CACHE_LIMIT})
        return parse_facets(payload, api_field)

    async def facets(self, field: str, query: str | None = None, limit: int = 25) -> FacetResult:
        values = await self._all_facet_values(field, query)
        truncated = len(values) > limit
        hint = (f"Showing the top {limit} of {len(values)} values. If the user asked for all of them, "
                f"call get_facets again with limit={min(len(values), 200)}.") if truncated else None
        return FacetResult(field=field, values=values[:limit], total_distinct=len(values),
                           truncated=truncated, hint=hint)

    async def facet_values(self, field: str) -> list[FacetValue]:
        """Every distinct value of a field across the whole catalogue (cached per process)."""
        if field not in self._facet_cache:
            values = [v for v in await self._all_facet_values(field)
                      if not _LEGACY_VALUE.match(v.name)]
            if not values:  # never cache an outage: every later lookup would wrongly fail
                raise WorldBankAPIError(
                    f"The list of valid {field.replace('_', ' ')} values is unavailable right now. "
                    "Try again shortly, or use search_documents with keywords instead.")
            self._facet_cache[field] = values
        return self._facet_cache[field]

    async def _alias(self, field: str, value: str) -> list[FacetValue] | None:
        """Catalogue values for a known alias ("Sub-Saharan Africa"), with real counts."""
        names = FIELD_ALIASES.get(field, {}).get(_norm(value))
        if not names:
            return None
        known = {v.name: v for v in await self.facet_values(field)}
        return [known[n] for n in names if n in known] or None

    async def resolve(self, field: str, terms: list[str], max_matches: int = 5) -> list[ResolveResult]:
        results = []
        for term in terms:
            alias = await self._alias(field, term)
            if alias:
                matches = [Match(value=v.name, count=v.count, score=1.0) for v in alias]
            else:
                matches = rank_candidates(term, await self.facet_values(field), max_matches)
            results.append(ResolveResult(field=field, term=term, matches=matches))
        return results

    async def _find_exact(self, field: str, value: str) -> FacetValue | None:
        wanted = _norm(value)
        return next((c for c in await self.facet_values(field) if _norm(c.name) == wanted), None)

    async def exact_value(self, field: str, value: str) -> str | None:
        """The catalogue's spelling of `value` if it exists exactly (ignoring case/punctuation)."""
        found = await self._find_exact(field, value)
        return found.name if found else None

    async def _with_dominant_variant(self, field: str, value: str, exact: FacetValue) -> list[str]:
        """Widen a rare exact spelling with the dominant official name ("Egypt" -> + "Egypt, Arab
        Republic of"); raise if several distinct values dominate ("Congo")."""
        words = _tokens(value)
        variants = [c for c in await self.facet_values(field)
                    if c.name != exact.name and words and words <= _tokens(c.name)
                    and c.count >= DOMINANT_RATIO * exact.count]
        if variants:
            biggest = max(v.count for v in variants)
            variants = [v for v in variants if v.count >= DOMINANT_SHARE * biggest]
        if len(variants) > 1:
            names = "; ".join(f"{v.name} ({v.count} docs)" for v in variants)
            raise WorldBankAPIError(
                f"'{value.strip()}' is ambiguous: {names}. Pass the specific {field} you mean.")
        return [variants[0].name, exact.name] if variants else [exact.name]

    async def normalize(self, field: str, value: str) -> list[str]:
        """Exact catalogue value(s) for `value` (case/spacing-insensitive, aliases expanded).

        Raises WorldBankAPIError with suggestions when nothing matches exactly, so the caller
        can self-correct instead of silently getting zero results.
        """
        alias = await self._alias(field, value)
        if alias:
            return [v.name for v in alias]
        exact = await self._find_exact(field, value)
        if exact:
            return await self._with_dominant_variant(field, value, exact)
        ranked = rank_candidates(value, await self.facet_values(field), 5)
        if ranked and ranked[0].score >= CONFIDENT_SCORE and (
                len(ranked) == 1 or round(ranked[0].score - ranked[1].score, 2) >= CONFIDENT_MARGIN):
            return [ranked[0].value]
        suggestions = [m.value for m in ranked]
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else (
            f" Call resolve_filter_values or get_facets with field='{field}' to see valid values.")
        hint += " Or put those words in `query` for a full-text match instead."
        raise WorldBankAPIError(f"No {field.replace('_', ' ')} named '{value.strip()}'.{hint}")
