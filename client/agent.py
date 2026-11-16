"""Agent client: answers questions about World Bank documents with a free LLM + the MCP server.

The client starts `server/wb_docs_mcp.py` as a subprocess (stdio), lists its tools, and runs a
tool-calling loop against any OpenAI-compatible chat endpoint (Ollama locally, or Groq):

    question -> LLM -> tool calls -> MCP server -> tool results -> LLM -> ... -> final answer

The full message history (including every tool call and result) is sent on each turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import mcp.types as mt
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # run as a script (python client/agent.py): make `client` importable
    sys.path.insert(0, str(PROJECT_ROOT))
from client.spinner import Spinner  # noqa: E402
SERVER_PATH = PROJECT_ROOT / "server" / "wb_docs_mcp.py"
MAX_STEPS = 8
# Chars of one tool result passed back to the LLM. Sized so a full 10-row page or the complete
# language list fits (server-side limits guarantee that), while protecting small context windows.
MAX_TOOL_TEXT = 9000

PROVIDERS: dict[str, dict[str, str]] = {
    # qwen2.5:3b with a 16k context window, built from ollama/Modelfile (see README).
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5-16k:3b"},
    # The task's example (llama-3.1-8b-instant) has been retired by Groq; gpt-oss-20b is free and
    # handles tool calling well. Override with LLM_MODEL (e.g. openai/gpt-oss-120b, qwen/qwen3.8-27b).
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b"},
}

SYSTEM_PROMPT = """\
You are a research assistant for World Bank documents. You can ONLY answer using the tools.

First decide which kind of question it is:
1. "What exists / which are most common" (document types, languages, topics, countries):
   call get_facets with the right field and answer from the counts.
   If the user asks for ALL values, call get_facets with limit=200.
2. "Find / list documents" about a country, region, topic, sector, document type or date
   range: call filter_documents. Pass the user's words (e.g. country="Brazil",
   sector="water and sanitation", region="Sub-Saharan Africa"); dates as YYYY-MM-DD.
   "Economic and sector work" and "project documents" are values of major_document_type.
   If it returns an error with suggestions, retry with a suggested value. Use
   resolve_filter_values when unsure. For plain keywords, call search_documents.
3. "What can you tell me about X / explain / why / how / what does the World Bank say":
   call search_documents (or filter_documents), then call get_document on the 2 most
   relevant results to read their abstracts. Answer the question itself in a few sentences
   or bullet points, using only what those documents say, then list the documents you used
   (title, date, URL).
4. Details of one specific document: call get_document.

Answer rules: use only facts from tool results, never outside knowledge. If the user asks
for your opinion, report what the World Bank documents say instead. Tool results are
already ordered: keep that order. Call results "most recent" only if they were sorted
newest first. For document lists, give title, date and URL, and mention the `total` number
of matches. If nothing was found, say so and stop: do not fill the gap from your own
knowledge. Keep the answer concise.

Format: the answer is printed in a plain terminal, so write plain text. No Markdown: no
**bold**, no # headings, no tables, no [text](url) links. Use short paragraphs and simple
"- " bullets, and write URLs out in full."""


@dataclass(frozen=True)
class Settings:
    provider: Literal["ollama", "groq"]
    model: str
    base_url: str
    api_key: str


def load_settings(env: Mapping[str, str]) -> Settings:
    """Read LLM_PROVIDER / LLM_MODEL / GROQ_API_KEY. Defaults to local Ollama qwen2.5-16k:3b (ollama/Modelfile)."""
    provider = (env.get("LLM_PROVIDER") or "ollama").strip().lower()
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama' or 'groq'.")
    defaults = PROVIDERS[provider]
    if provider == "groq":
        api_key = (env.get("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise SystemExit("LLM_PROVIDER=groq needs GROQ_API_KEY (free key: https://console.groq.com).")
    else:
        api_key = "ollama"  # Ollama ignores the key, but the OpenAI client requires one
    return Settings(provider=provider, model=(env.get("LLM_MODEL") or "").strip() or defaults["model"],
                    base_url=defaults["base_url"], api_key=api_key)


class ToolCaller(Protocol):
    """What the loop needs from an MCP session (mcp.ClientSession satisfies this)."""

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...


def mcp_tools_to_openai(tools: list[mt.Tool]) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function-calling tool specs."""
    return [{"type": "function", "function": {
        "name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}}
        for tool in tools]


def tool_result_to_text(result: mt.CallToolResult) -> str:
    text = "\n".join(c.text for c in result.content if isinstance(c, mt.TextContent))
    if result.is_error:
        text = f"ERROR: {text}"
    if len(text) > MAX_TOOL_TEXT:
        text = text[:MAX_TOOL_TEXT] + "\n...[truncated]"
    return text


@dataclass(frozen=True)
class ToolCall:
    """A tool call as returned by the LLM, normalised (some backends omit ids or pre-parse args)."""
    id: str
    name: str
    arguments: Any  # JSON string (usual) or an already-parsed dict

    def arguments_json(self) -> str:
        return self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)


def _tool_calls(message: Any, step: int) -> list[ToolCall]:
    return [ToolCall(id=c.id or f"call_{step}_{i}", name=c.function.name, arguments=c.function.arguments)
            for i, c in enumerate(getattr(message, "tool_calls", None) or [])]


def _parse_arguments(raw: Any) -> dict | str:
    """Tool arguments as a dict, or an error string for the LLM."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError) as exc:
        return f"ERROR: tool arguments were invalid JSON ({exc}). Send a valid JSON object."
    return parsed if isinstance(parsed, dict) else "ERROR: tool arguments must be a JSON object."


async def _execute(call: ToolCall, session: ToolCaller) -> str:
    """Run one tool call. Every failure becomes text for the LLM, never an exception."""
    arguments = _parse_arguments(call.arguments)
    if isinstance(arguments, str):
        return arguments
    try:
        return tool_result_to_text(await session.call_tool(call.name, arguments))
    except Exception as exc:  # noqa: BLE001 - surface any transport/server failure to the LLM
        return f"ERROR: calling {call.name} failed: {exc}"


REPROMPT = "Please give your final answer to the question now, based on the tool results above."
NO_ANSWER = "The model returned no answer. Try rephrasing the question, or run with -v to see the tool calls."


async def run_agent(question: str, session: ToolCaller, llm: Any, model: str, tools: list[dict],
                    max_steps: int = MAX_STEPS,
                    on_event: Callable[[str], None] | None = None) -> str:
    """Tool-calling loop. Returns the LLM's final text answer."""
    emit = on_event or (lambda _event: None)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question}]
    reprompted = False
    for step in range(max_steps):
        response = await llm.chat.completions.create(
            model=model, messages=messages, tools=tools or None, temperature=0)
        message = response.choices[0].message if response.choices else None
        content = ((message.content if message else None) or "").strip()
        calls = _tool_calls(message, step)
        messages.append({"role": "assistant", "content": content, **({"tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments_json()}}
            for c in calls]} if calls else {})})
        if not calls:
            if content:
                return content
            if reprompted:
                return NO_ANSWER
            reprompted = True  # small models sometimes reply empty once; nudge them a single time
            messages.append({"role": "user", "content": REPROMPT})
            continue
        for call in calls:
            emit(f"-> {call.name}({call.arguments_json()})")
            result = await _execute(call, session)
            emit(f"<- {result[:200]}{'...' if len(result) > 200 else ''}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return f"Stopped after {max_steps} tool-calling steps without a final answer."


async def answer(question: str, settings: Settings, llm: Any = None,
                 on_event: Callable[[str], None] | None = None,
                 server_logs: bool | None = None) -> str:
    """Start the MCP server over stdio, expose its tools to the LLM, and answer `question`.

    `server_logs` shows the server's stderr (default: only when `on_event` is set, i.e. tracing).
    """
    llm = llm or AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)],
                                   cwd=str(PROJECT_ROOT))
    show_logs = bool(on_event) if server_logs is None else server_logs
    with open(os.devnull, "w") as devnull:
        errlog = sys.stderr if show_logs else devnull
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = mcp_tools_to_openai((await session.list_tools()).tools)
                if on_event:
                    on_event(f"[{settings.provider}:{settings.model}] tools: "
                             + ", ".join(t["function"]["name"] for t in tools))
                return await run_agent(question, session, llm, settings.model, tools,
                                       on_event=on_event)


def _root_cause(exc: BaseException) -> BaseException:
    """anyio/MCP wrap failures in ExceptionGroups ("unhandled errors in a TaskGroup"); unwrap them."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _utf8_output() -> None:
    """Windows pipes/redirects default to cp1252, which can't encode characters models often emit."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> None:
    _utf8_output()
    parser = argparse.ArgumentParser(description="Ask questions about World Bank documents.")
    parser.add_argument("question", nargs="*", help="Question (omit for interactive mode)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each tool call and result to stderr")
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_settings(os.environ)
    trace = (lambda event: print(event, file=sys.stderr, flush=True)) if args.verbose else None

    async def answer_with_progress(question: str) -> str:
        # Verbose mode prints every step instead; the spinner only runs on a real terminal.
        spinner = Spinner(enabled=not args.verbose and sys.stderr.isatty())
        async with spinner:
            return await answer(question, settings, on_event=trace or spinner.on_event,
                                server_logs=args.verbose)

    def ask(question: str) -> None:
        try:
            print(asyncio.run(answer_with_progress(question)))
        except Exception as exc:  # noqa: BLE001 - CLI boundary: report and keep going
            cause = _root_cause(exc)
            print(f"Error: {type(cause).__name__}: {cause}. Is {settings.provider} running/configured? "
                  f"(model: {settings.model}, url: {settings.base_url})", file=sys.stderr)

    if args.question:
        ask(" ".join(args.question))
        return
    print(f"World Bank documents assistant ({settings.provider}: {settings.model}). Empty line to quit.")
    try:
        while question := input("\nQuestion> ").strip():
            ask(question)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
