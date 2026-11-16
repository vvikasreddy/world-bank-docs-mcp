# Design notes

Details behind the [README](../README.md): LLM choice, tool design, API quirks, testing, example runs and limitations.

## LLM choice

| | Ollama `qwen2.5:3b` (**default**) | Groq `openai/gpt-oss-20b` (optional) |
|---|---|---|
| Cost | Free, runs locally | Free tier (key from console.groq.com) |
| Limits | None | Daily / per-minute token caps |
| Privacy / offline | Fully offline | Cloud |
| Tool calling | Native, via OpenAI-compatible `/v1` | Native, via OpenAI-compatible `/v1` |
| Speed (this laptop, 4 GB GPU) | ~17–115 s per question | ~10 s per question |

**Groq model:** the task lists `llama-3.1-8b-instant` as the Groq example, but Groq has since
retired it (requests return "model does not exist"). The client defaults to `openai/gpt-oss-20b`,
which is free, available on a free key, and calls tools reliably. `openai/gpt-oss-120b` and
`qwen/qwen3.8-27b` also answered the example questions correctly (set `LLM_MODEL`).

**Where the time goes:** the World Bank API answers in ~0.4 s. Almost all of a question's time
is the LLM: loading the model (~9 s on the first call), re-reading the prompt each round (tool
definitions + history) and generating the answer token by token. Long answers (10 documents
with abstracts) take longest.

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
# .env  (git-ignored; never put keys in .env.example)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
# LLM_MODEL=openai/gpt-oss-120b   # optional override, works for either provider
```

Settings are read in this order: shell environment variables, then `.env`, then the built-in
defaults (Ollama `qwen2.5-16k:3b`). A Groq key alone does not switch providers;
`LLM_PROVIDER=groq` is needed too. With `-v`, the first line shows the provider and model in use.

**Groq's free tier has a per-minute token limit.** A question that needs 5–6 LLM rounds can hit
it; Groq then answers `429 Too Many Requests` and the client waits and retries automatically
(15–30 s per wait). A question that took ~10 s normally took over 100 s while throttled.

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
- No agent framework (LangChain etc.): the loop is ~30 lines on top of the MCP SDK client and the
  `openai` library, so every step is visible.

**System prompt (`SYSTEM_PROMPT` in `client/agent.py`):** tells the model which kind of question
it has (counts, document list, or explanation) and which tools to use for each; to use only facts
from tool results and say plainly when nothing was found; and to write plain terminal text with
no Markdown. Each tool's own description comes from its docstring in `server/wb_docs_mcp.py`.

**CLI details:** output is written as UTF-8 (Windows pipes default to cp1252, which can't encode
characters models often produce), and errors show their real cause instead of anyio's
"unhandled errors in a TaskGroup".

**Code layout:**

```
server/wb_api.py        HTTP, parsing, error mapping, facet cache + resolver (the only HTTP call site; no MCP code)
server/wb_docs_mcp.py   FastMCP app: 5 thin tools, shared validated param types, workflow prompt
client/agent.py         settings, MCP→OpenAI tool conversion, agent loop, stdio wiring, CLI
client/spinner.py       terminal progress line and tool-call lines
scripts/smoke_server.py calls every tool over stdio against the live API (no LLM needed)
scripts/run_examples.py runs the task's 6 suggested questions and writes docs/example_runs.md
tests/                  offline unit tests (API fixtures captured from the live service) + live contract tests
```

---

## Running and testing

```bash
pytest                         # 110 offline tests: parsing, errors, resolver, every tool, agent loop, CLI
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
All six suggested questions, with full tool traces, are in [`docs/example_runs.md`](example_runs.md).
These runs predate the plain-text rule in the system prompt, which is why they contain Markdown.

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
- **Speed:** 15–120 s per question with local inference on this laptop (4 GB GPU, part of the
  model runs on the CPU). Groq is ~10 s unless its free-tier rate limit kicks in.
- **No memory between questions.** Each question in interactive mode starts fresh, so follow-ups
  like "what about India?" don't know the previous question.
- **Formatting and relevance depend on the model.** The prompt asks for plain text and only
  relevant documents; the 3B model sometimes still uses `**bold**`, or lists loosely matching
  documents when nothing relevant exists (keyword search always returns its closest matches).
- **The model chooses the tools.** It usually follows the rules (filters for countries and
  document types, keywords for subjects) but not always: "GST in India" was sent as a keyword
  search instead of `country=India` + `query=GST`. Answers were still correct.
