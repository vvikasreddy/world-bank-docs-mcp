# World Bank Documents MCP Server + Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Native, chosen) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax. Every task follows superpowers:test-driven-development: RED (run, watch fail) → GREEN → full `pytest` → commit.

**Goal:** A local FastMCP server wrapping the World Bank Documents API, with discovery-first tools, exercised by a CLI agent running the free local Ollama model `qwen2.5:3b`.

**Architecture:** `server/wb_api.py` is the only code that talks HTTP (request helper, parsing, error mapping, facet-value resolution). `server/wb_docs_mcp.py` exposes thin FastMCP tools over it on stdio. The tools follow a **discover → resolve → retrieve → inspect** workflow: the LLM first learns which values exist (facets), maps the user's wording to exact World Bank values, then filters or searches, then drills into single documents. `client/agent.py` spawns the server with the MCP Python SDK stdio client and runs an OpenAI-compatible tool-calling loop (Ollama now; Groq is an env-var switch later).

**Tech Stack:** Python 3.13, `fastmcp` (server), `mcp` SDK (client: `ClientSession`, `stdio_client`, `StdioServerParameters`), `httpx`, `pydantic` v2, `openai` (OpenAI-compatible client), `python-dotenv`, `pytest` + `pytest-asyncio`. Ollama (already installed at `C:\Users\vikas\AppData\Local\Programs\Ollama`) with `qwen2.5:3b`.

**Spec:** `D:\Projects\world_bank_assessment\WB_MCP_Interview_Task.docx` (copied into `docs/` in Task 1).

**Context (why):** Interview take-home, graded on correctness 30%, MCP design 25%, code quality 20%, agentic loop 15%, README 10%. User direction: **build the server first, use the spec's Ollama model (`qwen2.5:3b`) for testing, actually exercise the server, and feel free to improve tools and agent architecture, e.g. "first get the topics, then act"**. A 3B model is weak at guessing exact filter values, so the server does the resolving work for it.

## Global Constraints

- Python (required); Python 3.13 at `C:\Python313`; venv in `.venv`.
- API base `https://search.worldbank.org/api/v3/wds`; always send `format=json`; no auth.
- Required tools present with exact names: `search_documents`, `filter_documents`, `get_document`, `get_facets`. Extra tools are allowed.
- Each tool: Pydantic-validated inputs, graceful API errors with informative messages, pagination where applicable, docstring covering inputs, outputs and when to use it.
- Client: stdio subprocess, MCP Python SDK client, question from CLI arg or interactive input, full message history, final answer printed to stdout.
- LLM now: Ollama `qwen2.5:3b` at `http://localhost:11434/v1` (default `LLM_PROVIDER=ollama`). Groq (`llama-3.1-8b-instant`, `https://api.groq.com/openai/v1`, `GROQ_API_KEY`) is supported via `LLM_PROVIDER=groq`; `LLM_MODEL` overrides the model.
- Deliverables: `server/wb_docs_mcp.py`, `client/agent.py`, `requirements.txt`, `README.md` (setup, LLM choice justified, 3 example questions **with real outputs**).
- Out of scope: auth, UI, Docker, observability.

## Verified API facts (live probe, 2026-11-14)

- Search response top level: `rows`, `os`, `page`, `total`, `documents`.
- `documents` is a dict keyed `"D<id>"`; entries have `id`, `display_title`, `docdt` (`"2021-07-01T04:00:00Z"`), `docty`, `count` (country), `lang`, `pdfurl`, `url`, `guid`, `abstracts: {"cdata!": "..."}`; some entries carry an empty `facets: {}`. When `fct` is requested, `documents["facets"]` holds the facets → **skip key `"facets"` when iterating**.
- Facets: `documents["facets"]["docty_exact"]` = `{"0": {"count": 144407, "name": "Procurement Plan", "label": "..."}, ...}`.
- To probe in Task 1: sorting `srt=docdt&order=desc`; id lookup param; nonexistent-id response; `rows=0` with `fct`; the full facet list size for `count_exact`/`topic_exact` (decides the facet fetch limit for resolution); region field (`admreg_exact` vs others) for "Sub-Saharan Africa".

## Tool design (discovery-first)

| # | Tool | Purpose | Key params → API |
|---|---|---|---|
| 1 | `get_facets` (required) | **Discover**: list distinct values + counts for a field. Answers "what types/languages exist?" | `field: FacetField`, `query?`, `limit 1–100 = 25` → `fct`, `qterm`, `rows=0` |
| 2 | `resolve_filter_values` (new) | **Resolve**: map the user's words to exact WB values (`"kenya"`→`"Kenya"`, `"water"`→ matching topics). Returns the best candidates with counts | `field: FacetField`, `terms: list[str] (1–5)`, `max_matches 1–10 = 5` |
| 3 | `filter_documents` (required) | **Retrieve** with exact filters + optional keywords, date range, newest-first sort | `country?`, `topic?`, `document_type?`, `region?`, `language?`, `start_date?`, `end_date?`, `query?`, `newest_first=True`, `rows`, `offset` → `*_exact`, `strdate`, `enddate`, `qterm`, sort |
| 4 | `search_documents` (required) | **Retrieve** by free text when no structured filter fits | `query`, `language?`, `rows`, `offset` → `qterm`, `lang_exact`, `rows`, `os` |
| 5 | `get_document` (required) | **Inspect** full metadata for one id | `document_id (^D?\d+$)` |

`FacetField = Literal["docty_exact","majdocty_exact","lang_exact","count_exact","admreg_exact","topic_exact","theme_exact"]` (adjust to what Task 1 confirms).

**Server-side safety net for small models:** before querying, `filter_documents` normalises each exact-filter value through the same resolver (a case- and whitespace-insensitive exact match against the facet list, cached per field). If a value has no exact match, the tool returns a `ToolError` listing the closest valid values ("No country 'Kenia'. Did you mean: Kenya?"). The LLM can then self-correct in one step instead of silently getting 0 results.

Resolver algorithm (`resolve_values` in `wb_api.py`): fetch the facet list once per field (`fct=<field>`, up to the max the API allows) and cache it in memory for the process. Per term, score candidates by: exact case-insensitive match (1.0) > candidate contains the term or vice versa (0.8) > `difflib.SequenceMatcher` ratio ≥ 0.6. Ties break by document count. Return the top `max_matches`.

## File Structure

```
server/__init__.py
server/wb_api.py          # HTTP, parsing, errors, output models, facet cache + resolver (no MCP imports)
server/wb_docs_mcp.py     # FastMCP app, 5 tools, shared Annotated param types, workflow prompt, main()
client/__init__.py
client/agent.py           # settings, tool conversion, run_agent loop, stdio wiring, CLI
tests/fixtures/*.json     # real API responses captured in Task 1
tests/test_wb_api.py  tests/test_resolver.py  tests/test_server_tools.py  tests/test_agent.py
tests/test_live.py        # @pytest.mark.live (real API / real Ollama), skipped by default
pytest.ini requirements.txt .env.example .gitignore README.md
docs/WB_MCP_Interview_Task.docx  docs/example_runs.md  docs/superpowers/plans/2026-11-14-wb-docs-mcp.md
```

## Review Focus

1. **User wording ≠ exact WB value** ("kenya", "Sub-Saharan Africa", "water and sanitation") → resolver maps it, or `filter_documents` errors with suggestions. Tests: Task 3 `test_resolve_*`; Task 4 `test_filter_unknown_value_suggests`.
2. **LLM sends invalid args** (`start_date="2019"`, `rows=100`) → validation error returned as tool output; loop continues. Tests: Task 4 `test_filter_rejects_bad_date`; Task 6 `test_tool_error_is_fed_back_to_llm`.
3. **Zero results / offset past the end** → empty list, `next_offset=None`, plus a `hint`; no exception. Test: Task 2 `test_parse_search_empty`.
4. **Context blow-up on a 3B model** (long abstracts × rows) → abstracts ≤300 chars, rows ≤20 (default 5), client caps tool text at 6000 chars. Test: Task 2 `test_abstract_truncated`.
5. **Small model loops or never answers** → `max_steps=8`, then a clear stop message. Test: Task 6 `test_stops_after_max_steps`.

---

## Phase A: MCP server (build first)

### Task 1: Scaffold, dependencies, Ollama model, live API probe, fixtures

- [ ] `git init`; `.gitignore` (`.venv/ __pycache__/ .env .pytest_cache/`); copy the spec and this plan into `docs/`.
- [ ] `python -m venv .venv`; install `fastmcp mcp httpx pydantic openai python-dotenv pytest pytest-asyncio`; write `requirements.txt` with `>=` floors from the installed versions.
- [ ] Verify imports: `from fastmcp import FastMCP, Client; from fastmcp.exceptions import ToolError; from mcp import ClientSession, StdioServerParameters; from mcp.client.stdio import stdio_client`. If the installed `mcp` v2 moved these, pin `mcp<2` (and a compatible `fastmcp`) and note it in the README.
- [ ] `ollama pull qwen2.5:3b` (~2 GB, runs in the background); verify with `ollama list`, then a one-shot tool-call request to `http://localhost:11434/v1/chat/completions` to confirm tool calling works.
- [ ] `pytest.ini`: `asyncio_mode = auto`, marker `live`, `addopts = -m "not live"`.
- [ ] Probe the API and save fixtures `tests/fixtures/`: `search_climate_kenya.json` (`qterm=climate Kenya&rows=3`), `facets_docty.json` (`fct=docty_exact&rows=0`), `facets_country.json` (`fct=count_exact&rows=0`), `get_document.json`, `empty.json` (`qterm=zzqqxxnonexistent`), `sorted_indonesia.json` (`count_exact=Indonesia&srt=docdt&order=desc&rows=3`). Record the findings (id param, sort works?, facet list size, region field) as a module comment at the top of `wb_api.py`.
- [ ] `.env.example`: `LLM_PROVIDER=ollama`, `LLM_MODEL=`, `GROQ_API_KEY=`.
- [ ] Commit `chore: scaffold project, deps, API fixtures`.

### Task 2: `server/wb_api.py`, the API client and parsing

**Produces:** `WorldBankAPIError(Exception)`; Pydantic models `DocumentSummary(id, title, date, doc_type, country, language, abstract, pdf_url, url)` (all `str | None` except `id`, `title`), `SearchResult(total: int, offset: int, rows: int, next_offset: int | None, documents: list[DocumentSummary], hint: str | None = None)`, `FacetValue(name: str, count: int)`, `FacetResult(field: str, values: list[FacetValue])`; `ABSTRACT_MAX_CHARS = 300`; `parse_documents(payload) -> list[DocumentSummary]`; `parse_facets(payload, field, limit) -> FacetResult`; `class WorldBankClient(http: httpx.AsyncClient | None = None, timeout: float = 20.0)` with `fetch(params) -> dict` (the only HTTP call site; adds `format=json`, drops `None`), `search(params, offset, rows) -> SearchResult`, `get_document(doc_id) -> dict`, `facets(field, query, limit) -> FacetResult`.

- [ ] **RED** `tests/test_wb_api.py` (HTTP faked via `httpx.MockTransport`): `test_parse_documents_skips_facets_key`; `test_parse_documents_maps_fields` (country "Kenya", date `^\d{4}-\d{2}-\d{2}$`, pdf_url http…); `test_abstract_truncated` (≤301 chars, ends "…"); `test_parse_search_empty`; `test_next_offset` (50/0/5→5; 7/5/5→None); `test_parse_facets_sorted_and_limited` (limit 3, first "Procurement Plan"); `test_fetch_timeout_raises_api_error` ("timed out"); `test_fetch_http_error_raises_api_error` ("503"); `test_fetch_non_json_raises_api_error`; `test_fetch_sends_format_json_and_drops_none`; `test_get_document_not_found` ("No document found").
- [ ] Run → FAIL. **GREEN**: implement. Full `pytest` → PASS. Commit `feat(server): World Bank API client`.

### Task 3: Facet cache + resolver

**Produces (in `wb_api.py`):** `Match(value: str, count: int, score: float)` model; `ResolveResult(field: str, term: str, matches: list[Match])`; pure `rank_candidates(term: str, candidates: list[FacetValue], max_matches: int) -> list[Match]`; `WorldBankClient.facet_values(field) -> list[FacetValue]` (cached per client instance); `WorldBankClient.resolve(field, terms, max_matches) -> list[ResolveResult]`; `WorldBankClient.normalize(field, value) -> str` (returns the exact facet name, or raises `WorldBankAPIError` with up to 5 suggestions).

- [ ] **RED** `tests/test_resolver.py`: `test_resolve_case_insensitive_exact` ("kenya"→"Kenya", score 1.0); `test_resolve_substring` ("water" hits a topic containing "Water"); `test_resolve_typo` ("Kenia"→"Kenya"); `test_resolve_ranks_ties_by_count`; `test_resolve_no_match_returns_empty`; `test_facet_values_cached` (MockTransport call count == 1 after two resolves); `test_normalize_unknown_suggests` (message contains "Did you mean" and "Kenya").
- [ ] Run → FAIL. **GREEN**: implement with `difflib`. Full `pytest` → PASS. Commit `feat(server): facet-value resolver`.

### Task 4: `server/wb_docs_mcp.py`, the FastMCP server with 5 tools

**Produces:** `mcp = FastMCP("worldbank-docs", instructions=WORKFLOW)`; module-level `api = WorldBankClient()` (tests monkeypatch it); the 5 tools from the table above; an MCP prompt `research_workflow` describing discover → resolve → retrieve → inspect; `main()` → `mcp.run()`.

Validation: shared `Annotated` aliases defined once (`Rows` 1–20 default 5, `Offset ≥0`, `IsoDate` pattern `^\d{4}-\d{2}-\d{2}$`, `FacetField` Literal). FastMCP builds the Pydantic model and JSON schema from them. Flat args suit 3B models better than nested objects. Cross-field rules (at least one filter, start ≤ end) raise `ToolError`. One helper `_api_call(coro)` maps `WorldBankAPIError` → `ToolError` for every tool. Docstrings state "when to use" and point to the next workflow step.

- [ ] **RED** `tests/test_server_tools.py` (in-memory `fastmcp.Client(mcp)`; `api` replaced by a `WorldBankClient` on a fixture-serving MockTransport that records params):
  - `test_lists_required_and_extra_tools` (the 4 required ⊆ names; `resolve_filter_values` present; every description > 40 chars).
  - `test_search_maps_params` (qterm, lang_exact, rows=3, os=6).
  - `test_search_rejects_rows_over_20`.
  - `test_filter_requires_at_least_one_filter`.
  - `test_filter_rejects_bad_date`; `test_filter_rejects_start_after_end`.
  - `test_filter_normalizes_values` ("kenya" → request `count_exact=Kenya`).
  - `test_filter_unknown_value_suggests` ("Kenia" → error contains "Kenya").
  - `test_filter_newest_first_sends_sort`.
  - `test_filter_zero_results_includes_hint`.
  - `test_get_document_accepts_D_prefix`.
  - `test_get_facets_rejects_unknown_field`.
  - `test_resolve_filter_values_tool` (terms ["kenya","brazil"] → 2 results).
  - `test_api_failure_becomes_tool_error` (500 → "World Bank API").
- [ ] Run → FAIL. **GREEN**: implement. Full `pytest` → PASS.
- [ ] **Try it yourself:** `npx @modelcontextprotocol/inspector .venv\Scripts\python server\wb_docs_mcp.py` and call every tool against the live API; plus `scripts/smoke_server.py`, which drives the server over stdio with `fastmcp.Client("server/wb_docs_mcp.py")` and prints one call per tool. Fix anything odd via TDD.
- [ ] Commit `feat(server): FastMCP server with discovery-first tools`.

### Task 5: Live API contract tests

- [ ] `tests/test_live.py` (`@pytest.mark.live`): search climate Kenya total > 0; `lang_exact` facets contain "English"; get_document on an id from search has `display_title`; Indonesia newest-first dates non-increasing; resolver maps "kenya" → "Kenya" against the real facets.
- [ ] `pytest -m live -v` → PASS. Commit `test: live API contract tests`.

## Phase B: Agent client (Ollama) to exercise the server

### Task 6: `client/agent.py`, the agent loop (pure, injectable)

**Produces:** `Settings(provider, model, base_url, api_key)` + `load_settings(env) -> Settings` (ollama default: `qwen2.5:3b`, `http://localhost:11434/v1`, key `"ollama"`; groq: needs `GROQ_API_KEY` else `SystemExit` with instructions); `mcp_tools_to_openai(tools) -> list[dict]`; `ToolCaller` Protocol (`call_tool(name, arguments)`); `tool_result_to_text(result) -> str` (`"ERROR: "` prefix when `isError`, 6000-char cap); `async run_agent(question, session, llm, model, tools, max_steps=8) -> str`; `SYSTEM_PROMPT`.

`SYSTEM_PROMPT` encodes the workflow explicitly for the small model: (1) for "what exists / most frequent" questions → `get_facets`; (2) before filtering on a country, topic, region or document type → `resolve_filter_values`; (3) retrieve with `filter_documents` (or `search_documents` for pure keywords); (4) `get_document` only when details are needed; (5) answer only from tool results, citing title, date and URL; say plainly when nothing was found.

Loop: `messages=[system, user]`; each step `chat.completions.create(model, messages, tools, tool_choice="auto")`; append the assistant message (with `tool_calls`); no tool calls → return the content; else for each call parse the JSON args (bad JSON → error text back to the model), `await session.call_tool`, append `{"role":"tool","tool_call_id","content"}`. Past `max_steps` → `"Stopped after 8 tool-calling steps without a final answer."`.

- [ ] **RED** `tests/test_agent.py` (`FakeLLM` scripted in the OpenAI response shape that records the messages it receives; `FakeSession`): `test_returns_direct_answer_without_tools`; `test_executes_tool_call_and_feeds_result_back` (matching `tool_call_id`); `test_keeps_full_history_across_multiple_tool_rounds`; `test_handles_multiple_tool_calls_in_one_turn`; `test_tool_error_is_fed_back_to_llm`; `test_malformed_tool_arguments_reported_not_raised`; `test_stops_after_max_steps`; `test_mcp_tools_to_openai_shape`; `test_load_settings_ollama_default`; `test_load_settings_groq_requires_key`.
- [ ] Run → FAIL. **GREEN**: implement. Full `pytest` → PASS. Commit `feat(client): agentic tool-calling loop`.

### Task 7: stdio wiring, CLI, real runs with Ollama

**Produces:** `async answer(question, settings, llm=None) -> str` (`stdio_client(StdioServerParameters(command=sys.executable, args=[SERVER_PATH]))` → `ClientSession` → `initialize` → `list_tools` → `run_agent`); `main(argv=None)` (question from the CLI arg, else an interactive loop; loads `.env`; prints the answer; `--verbose` prints each tool call and its args to stderr so the workflow is visible).

- [ ] **RED** `test_live.py::test_answer_end_to_end_stdio` (live): real server subprocess + `FakeLLM` scripted to call `get_facets(field="lang_exact", limit=5)` → the tool message contains "English".
- [ ] Run → FAIL; implement; → PASS. Full `pytest` → PASS.
- [ ] **Real runs on `qwen2.5:3b`:** all 6 spec §7 questions with `--verbose`; log the questions, tool traces and answers to `docs/example_runs.md`. Where the model picks badly, tune the tool docstrings and `SYSTEM_PROMPT` (tests stay green). Record honest observations about 3B behaviour for the README.
- [ ] Commit `feat(client): stdio MCP wiring, CLI, example runs`.

### Task 8: README + quality pass

- [ ] `README.md`: what it is; architecture diagram (text) and the discover → resolve → retrieve → inspect workflow; tool reference table; setup (venv, requirements, `ollama pull qwen2.5:3b`, `.env`); LLM choice and justification (spec-recommended, free, fully offline, no rate limits; Groq switch documented for faster/stronger runs); running the server, Inspector, smoke script, client and tests (`pytest`, `pytest -m live`); design decisions (flat Annotated params, server-side value normalisation for small models, compact outputs, error strategy); **3 example questions with real outputs** from `docs/example_runs.md`; known limitations.
- [ ] Run `/ecc:python-review`, then the `ecc:silent-failure-hunter` and `ecc:code-simplifier` agents on `server/` and `client/`; apply valid findings via TDD.
- [ ] Final: `pytest` + `pytest -m live` green; fresh-venv smoke run of one question. Commit `docs: README and example outputs`.

## Verification (end-to-end)

1. `pytest` → all offline tests green; `pytest -m live` → live API, resolver and stdio E2E green.
2. MCP Inspector and `scripts/smoke_server.py` show all 5 tools returning real data.
3. `python client/agent.py --verbose "What document types does the World Bank publish most frequently?"` on `qwen2.5:3b` → trace shows `get_facets`; the answer lists the top document types with counts.
4. `python client/agent.py --verbose "Find project documents related to water and sanitation in Brazil."` → trace shows `resolve_filter_values` → `filter_documents`; the answer cites real titles and URLs.
5. README examples are pasted from real runs.
