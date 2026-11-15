# World Bank Documents — MCP Server + Agent

A local **MCP server** that wraps the public [World Bank Documents & Reports API](https://documents.worldbank.org/en/publication/documents-reports/api),
plus an **agent client** that lets a free, local LLM (**Ollama `qwen2.5:3b`**) answer natural-language
questions grounded in real World Bank publications.

```
you ──question──▶ client/agent.py ──chat + tool specs──▶ LLM (Ollama qwen2.5:3b / Groq)
                        ▲   │                                   │
                        │   │◀──────── tool calls ──────────────┘
                        │   ▼
                        │  MCP ClientSession ══ stdio ══▶ server/wb_docs_mcp.py (FastMCP)
                        │                                       │
                        └──── final answer ◀── tool results ◀───┴──▶ search.worldbank.org/api/v3/wds
```

---

## Quick start

Requires **Python 3.11+** (developed on 3.13) and [Ollama](https://ollama.com).

```bash
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. The free local model (~1.9 GB, one-time), rebuilt with a 16k context window
ollama pull qwen2.5:3b
ollama create qwen2.5-16k:3b -f ollama/Modelfile    # reuses the same weights, no new download

# 3. (optional) configuration — defaults work out of the box
copy .env.example .env            # macOS/Linux: cp .env.example .env

# 4. Ask a question
python client/agent.py "What document types does the World Bank publish most frequently?"
python client/agent.py -v "Find project documents related to water and sanitation in Brazil."   # -v shows each tool call
python client/agent.py            # interactive mode (empty line quits)
```

The client starts the MCP server itself (as a stdio subprocess), so there is nothing else to run.

---

## LLM choice

| | Ollama `qwen2.5:3b` (**default**) | Groq `llama-3.1-8b-instant` (optional) |
|---|---|---|
| Cost | Free, runs locally | Free tier (key from console.groq.com) |
| Limits | None | Daily / per-minute token caps |
| Privacy / offline | Fully offline | Cloud |
| Tool calling | Native, via OpenAI-compatible `/v1` | Native, via OpenAI-compatible `/v1` |
| Speed (this laptop, CPU) | ~15–60 s per question | ~1–3 s per question |

**Why Ollama + `qwen2.5:3b`:** it is the task's recommended default, costs nothing, has no
rate limits (useful while iterating on tool descriptions) and runs completely offline. It is also
a *hard* target: a 3B model is easily confused by exact-match filters, which pushed the server
design toward being forgiving (see [Design decisions](#design-decisions)). Anything that works
on a 3B model works better on a larger one.

**Why the 16k rebuild (`ollama/Modelfile`):** Ollama runs `qwen2.5:3b` with a **4,096-token**
context by default (`ollama ps` shows it). The 5 tool definitions (~1.8k tokens), the system
prompt and one tool result go over that. Ollama then **silently drops the oldest messages,
starting with the system prompt**, and Ollama's OpenAI-compatible endpoint ignores a `num_ctx`
option. The Modelfile is just `FROM qwen2.5:3b` + `PARAMETER num_ctx 16384`. To use the stock
model anyway: `LLM_MODEL=qwen2.5:3b`.

**Switching to Groq** needs no code changes. Both backends expose the same OpenAI-compatible API:

```ini
# .env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
# LLM_MODEL=llama-3.3-70b-versatile   # optional override, works for either provider
```

---

## MCP tools

The four required tools are present with the required names, plus one extra discovery tool.
They are designed around a **discover → resolve → retrieve → inspect** workflow, which the server
also advertises to clients through its MCP `instructions` and a `research_workflow` prompt.

| Step | Tool | What it does | Key API params |
|---|---|---|---|
| Discover | `get_facets` | Distinct values of a field with counts, most frequent first. Answers "what types / languages exist?" | `fct`, `qterm`, `facet.limit` |
| Resolve | `resolve_filter_values` *(extra)* | Maps the user's wording to exact catalogue values: `kenya`→`Kenya`, `water and sanitation`→`Water, Sanitation and Waste Management`, `Sub-Saharan Africa`→its regions | `fct` (cached) |
| Retrieve | `filter_documents` | Exact filters: country, region, topic, sector, document type, major document type, language, date range, optional keywords | `count_exact`, `admreg_exact`, `teratopic_exact`, `sectr_exact`, `docty_exact`, `majdocty_exact`, `lang_exact`, `strdate`, `enddate`, `qterm`, `srt`/`order` |
| Retrieve | `search_documents` | Full-text search with optional language filter | `qterm`, `lang_exact` |
| Inspect | `get_document` | Key metadata for one document id (`33295056` or `D33295056`): title, date, type, country, project, authors, topics, abstract, links | `id` |

Every tool:
- **Validates inputs with Pydantic.** Parameters use shared `Annotated[..., Field(...)]` types,
  which FastMCP compiles into a Pydantic model and the published JSON schema:
  - `Rows` 1–10, `Offset` 0–10,000
  - `IsoDate` is a real `datetime.date`, so `2022-02-30` is rejected
  - a `Literal` of facet fields

  Cross-field rules such as "at least one filter" and `start_date ≤ end_date` return clear tool
  errors.
- **Handles API errors gracefully.**
  - Timeouts and 5xx errors are retried once, then reported as *"The World Bank API timed out
    after 20s. Try again or narrow the query."*
  - 4xx errors say the request itself was rejected, so retrying won't help.
  - Unreachable hosts, non-JSON bodies and malformed records each get their own readable
    `ToolError`, which the LLM can react to.
- **Supports pagination** (`rows` + `offset`). Results return `total` and `next_offset`, and a
  hint when `offset` is past the end.
- **Returns typed, compact output** (Pydantic models for every tool).
  - List abstracts are trimmed to 200 characters, and `get_document` returns a chosen subset of
    fields (~2 KB) instead of the raw 11 KB record.
  - So a full 10-row page or the complete language list always fits the agent's per-result
    budget, and nothing is cut mid-JSON.

---

## Design decisions

**Server-side value resolution, built for small models.** Probing the live API showed that
`*_exact` filters are **case-sensitive** (`count_exact=kenya` → 0 hits, `Kenya` → 9,738) and that
the catalogue's names rarely match how people speak ("water and sanitation" is
`Water, Sanitation and Waste Management`; "Sub-Saharan Africa" is two administrative regions).
So `filter_documents` normalises every filter value against the cached facet list:

- **Exact matches** (ignoring case and punctuation) are used. If the catalogue also has a
  *dominant* spelling of the same name, the two are combined.
  - The country list holds old and duplicate spellings, so a bare exact match can be badly wrong:
    "Egypt" matches a 14-document entry, while "Egypt, Arab Republic of" has 4,381.
  - So "Egypt" becomes `Egypt, Arab Republic of` OR `Egypt`.
  - If several dominant values compete ("Congo" → DRC 5,755 / Republic of Congo 2,514), the tool
    reports the value as **ambiguous** and lists them.
- **Confident matches** are applied automatically only when *all* of the user's words appear in
  the value **and** it clearly beats the runner-up. Results report `applied_filters`, so the
  substitution is visible.
- **Anything else** returns an error with suggestions, e.g. *"No country named 'Kenia'. Did you
  mean: Kenya? Or put those words in `query` for a full-text match instead."* The model corrects
  itself on the next step instead of silently getting zero results.
- **Small alias tables** cover names the catalogue spells differently (`Sub-Saharan Africa`,
  `MENA`, `Vietnam` → `Viet Nam`, `Turkey` → `Turkiye`, …). Aliases report real document counts.
- **Major types given as `document_type`:** a value that is really a *major* document type
  (`Project Documents`) is moved to `major_document_type`.
- **Unavailable value lists** are reported as an error and never cached. A temporary API problem
  can't make every later lookup fail.
- **`search_documents`'s `language`** goes through the same resolver ("english" → `English`).

**API quirks found by probing (documented in `server/wb_api.py`):**
- `topic_exact` matches nothing, so topics map to `teratopic_exact`.
- `documents` also contains a `facets` key, which is skipped when reading documents.
- Facet lists stop at 100 values unless you pass `facet.limit` (there are 488 countries and 198
  languages). `get_facets` always fetches the full list, so `total_distinct` and `truncated` are
  accurate, and it can return up to 200 values.
- Multiple values are OR-ed with `^`.
- Some sector values are legacy duplicates (`FY17 - …`) and are ignored when resolving.

**Sorting:** `filter_documents` sorts newest first when there is no keyword `query`, and by
relevance when there is one. Sorting keyword searches by date buried the relevant results.

**Forgiving inputs:** a single string is accepted where a list is expected, and explicit `null`s
are accepted. Both are common with small models.

**Agent loop (`client/agent.py`):**
- It keeps the **full message history**, so each turn sends the system prompt, the question,
  every assistant tool call and every tool result.
- It runs all tool calls in a turn.
- Odd LLM output is handled rather than crashing the loop:
  - malformed JSON arguments, or arguments that arrive already parsed as a dict
  - missing tool-call ids (an id is generated and reused, so strict backends accept the history)
  - an empty reply (the model is nudged once, then gets a clear "no answer" message)
  - empty `choices`
- Tool errors and session exceptions are fed back to the model as text.
- Each tool result is capped at 9,000 characters as a safety net. The server-side limits keep
  real results under it. The loop stops after `max_steps=8`.
- The loop is a pure function over an injected LLM and session, so it is unit-tested with fakes.

**Code layout:**

```
server/wb_api.py        HTTP, parsing, error mapping, facet cache + resolver (the only HTTP call site; no MCP code)
server/wb_docs_mcp.py   FastMCP app: 5 thin tools, shared validated param types, workflow prompt
client/agent.py         settings, MCP→OpenAI tool conversion, agent loop, stdio wiring, CLI
scripts/smoke_server.py calls every tool over stdio against the live API (no LLM needed)
scripts/run_examples.py runs the task's 6 suggested questions and writes docs/example_runs.md
tests/                  offline unit tests (API fixtures captured from the live service) + live contract tests
```

---

## Running and testing

```bash
pytest                         # 95 offline tests: parsing, errors, resolver, every tool, agent loop
pytest -m live                 # 10 tests against the real API, incl. a stdio end-to-end run
python scripts/smoke_server.py # exercise all 5 tools against the live API, no LLM involved
python scripts/run_examples.py # run the 6 suggested questions through the agent

# Inspect the server interactively with the official MCP Inspector:
npx @modelcontextprotocol/inspector .venv/Scripts/python server/wb_docs_mcp.py
```

The tests were written test-first. The offline suite uses real responses captured from the API
(`tests/fixtures/`) served through `httpx.MockTransport`, and calls the tools through FastMCP's
in-memory client, the same protocol path a real client uses.

---

## Example questions and outputs

Real runs on `ollama:qwen2.5-16k:3b` (laptop, partly GPU), copied verbatim apart from trimming.
All six suggested questions, with full tool traces, are in [`docs/example_runs.md`](docs/example_runs.md).

### 1. "What document types does the World Bank publish most frequently?"

```text
-> get_facets({"limit":100,"field":"document_type"})
<- {"field":"document_type","values":[{"name":"Procurement Plan","count":144407},{"name":"Implementation Status and Results Report","count":59904}, ...
```

> The most frequently published document types by the World Bank include "Procurement Plan",
> "Implementation Status and Results Report", "Auditing Document", "Working Paper", and "Brief".
> There are a total of 257 distinct document types published, with the full list truncated due to
> the limit of 100. If the user needs to see all the document types, they can request it again
> with a limit of 200.

### 2. "Find the most recent economic sector work on Indonesia."

The model follows the **resolve → retrieve** workflow by itself:

```text
-> resolve_filter_values({"field":"major_document_type","terms":["Economic & Sector Work"]})
<- [{"field":"major_document_type","term":"Economic & Sector Work","matches":[{"value":"Economic & Sector Work","count":14365,"score":1.0}]}]
-> filter_documents({"country":"Indonesia","major_document_type":"Economic & Sector Work","newest_first":true})
<- {"total":384,"offset":0,"rows":5,"next_offset":5,"documents":[{"id":"40131270","title":"Indonesia - Country Private Sector Diagnostic","date":"2026-10-05", ...
```

> The most recent document in the category of Economic and Sector Work on Indonesia is:
> - **Title:** Indonesia - Country Private Sector Diagnostic
> - **Date:** 2026-10-05
> - **Type:** Country Private Sector Diagnostics
> - **Abstract:** Indonesia has achieved rapid economic growth and substantial poverty reduction over the past five decades, driven significantly by the private sector. However, growth has slowed as investment in the private sector has not kept pace with the ove…
> - **PDF:** https://documents.worldbank.org/curated/en/099753310052628862/pdf/IDU-2c04a71f-c12a-4c70-b9f9-a1111c195bc1.pdf
> - **URL:** http://documents.worldbank.org/curated/en/099753310052628862

### 3. "Find documents about education financing in Sub-Saharan Africa published between 2019 and 2022."

The region alias, the date range and keywords are combined in one call:

```text
-> filter_documents({"region":["Sub-Saharan Africa"],"start_date":"2019-01-01","end_date":"2022-12-31","query":"education financing"})
<- {"total":11823,"offset":0,"rows":5,"next_offset":5,"documents":[{"id":"32415641","title":"Project Information Document-Integrated Safeguards Data Sheet - Additional Financing to the Education Reform S...
```

> I found 5 documents related to your query. Here are the details of the first 5 documents:
>
> 1. **Title:** Project Information Document-Integrated Safeguards Data Sheet - Additional Financing to the Education Reform Support Project - P172885
>    - **Date:** 2020-09-18 · **Country:** Cameroon · **URL:** http://documents.worldbank.org/curated/en/168871600414843039
> 2. **Title:** Project Information Document-Integrated Safeguards Data Sheet - Additional Financing to the Education Reform Support Project - P172885
>    - **Date:** 2020-05-06 · **Country:** Cameroon · **URL:** http://documents.worldbank.org/curated/en/241821588792389607
>
> *(3 more results in `docs/example_runs.md`)*

The answer says "5 documents" although the tool reported `total: 11823`. Summaries like this
are where a 3B model is weakest.

---

## Known limitations

- **A 3B model is the weak link, not the tools.**
  - It sometimes ignores hints. For "list **all** languages" it often keeps the default 25 values,
    even though the result says `truncated: true` and suggests `limit=198`.
  - It can misstate totals (example 3).
  - It can garble URLs when it rewrites them. The tool trace (`-v`) always shows the exact values.
  - With Groq or a larger model (`LLM_MODEL=qwen2.5:7b`) these problems mostly go away.
- **Topic search is coarse.** The API's topic field (`teratopic_exact`) has only 29 broad topics, so
  specific subjects ("climate resilience") work better as `query` keywords, which the model
  generally chose on its own in the example runs.
- **The alias tables are small.** They cover common names only; other wording goes through the
  fuzzy resolver.
- **Each question starts a fresh server process**, so the facet cache (used for resolving values)
  is refetched per question. That costs about 0.5 s per field.
- **Speed:** 15–120 s per question with local inference on this laptop.
