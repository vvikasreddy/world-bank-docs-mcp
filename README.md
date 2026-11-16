# World Bank Documents MCP

Ask questions in plain English and get answers from real World Bank publications. An **MCP
server** wraps the public [World Bank Documents & Reports API](https://documents.worldbank.org/en/publication/documents-reports/api),
and a small **agent** lets a free LLM call its tools.

## Quick start

Needs **Python 3.11+** and [Ollama](https://ollama.com).

```bash
python -m venv .venv
.venv\Scripts\activate                                # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:3b                                # free local model, ~1.9 GB, one-time
ollama create qwen2.5-16k:3b -f ollama/Modelfile      # same model with a 16k context window

python client/agent.py "What document types does the World Bank publish most frequently?"
```

The agent starts the MCP server for you.

```bash
python client/agent.py                 # interactive: one question after another
python client/agent.py -v "..."        # show every tool call and result
```

**Faster, free cloud model:** get a key at [console.groq.com](https://console.groq.com), copy
`.env.example` to `.env`, and set `LLM_PROVIDER=groq` and `GROQ_API_KEY=gsk_...`
(default Groq model: `openai/gpt-oss-20b`).

**MCP server on its own** (for any MCP client):

```bash
python server/wb_docs_mcp.py                                         # stdio
npx @modelcontextprotocol/inspector python server/wb_docs_mcp.py     # browse and call the tools
```

## How it works

```
you ─▶ client/agent.py ─▶ LLM (Ollama / Groq)
            │  ▲               │ tool calls
            ▼  │ results       ▼
       MCP over stdio ─▶ server/wb_docs_mcp.py ─▶ World Bank API
```

The client sends the question and the tool list to the LLM, runs the tools it asks for through
MCP, adds the results to the conversation, and repeats until the LLM writes the answer.

| Tool | Use |
|---|---|
| `get_facets` | What values exist and how common they are (document types, languages, countries…) |
| `filter_documents` | Documents by country, region, topic, sector, type, language, dates, plus keywords |
| `search_documents` | Plain keyword search |
| `get_document` | Full details of one document |
| `resolve_filter_values` | Maps user wording to the catalogue's exact names (`Vietnam` → `Viet Nam`) |

Every tool validates its inputs with Pydantic, returns clear errors, and pages results
(`rows`, `offset`, `total`, `next_offset`). `filter_documents` fixes the user's wording itself,
so the small model doesn't have to know exact catalogue names.

## Tests

```bash
pytest              # 110 offline tests (recorded API responses, fake LLM)
pytest -m live      # 10 tests against the real World Bank API
```

## More

- [docs/design.md](docs/design.md): LLM choice, tool and API details, design decisions,
  example outputs, known limitations
- [docs/example_runs.md](docs/example_runs.md): the task's six questions with full tool traces
