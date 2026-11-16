"""Tests for the agent loop, with a scripted fake LLM and fake MCP session (no network)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import mcp.types as mt
import pytest

from client.agent import (
    MAX_TOOL_TEXT,
    load_settings,
    mcp_tools_to_openai,
    run_agent,
    tool_result_to_text,
)


def tool_call(call_id: str, name: str, args) -> SimpleNamespace:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


def reply(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    """Mimics openai.AsyncOpenAI: returns scripted replies, records every request's messages."""

    def __init__(self, script: list):
        self.script = list(script)
        self.requests: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        return self.script.pop(0)


class FakeSession:
    def __init__(self, results: dict[str, mt.CallToolResult] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.results = results or {}

    async def call_tool(self, name: str, arguments: dict | None = None):
        self.calls.append((name, arguments))
        return self.results.get(name) or ok(f"{name} result")


def ok(text: str) -> mt.CallToolResult:
    return mt.CallToolResult(content=[mt.TextContent(type="text", text=text)], is_error=False)


async def run(llm, session, **kw) -> str:
    return await run_agent("What languages exist?", session, llm, "test-model", tools=[], **kw)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """main() loads the developer's .env; keep tests independent of it."""
    from client import agent
    monkeypatch.setattr(agent, "load_dotenv", lambda *a, **k: None)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def tool_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["role"] == "tool"]


async def test_returns_direct_answer_without_tools():
    llm = FakeLLM([reply("Hello")])
    assert await run(llm, FakeSession()) == "Hello"


async def test_executes_tool_call_and_feeds_result_back():
    llm = FakeLLM([reply(tool_calls=[tool_call("c1", "get_facets", {"field": "language"})]),
                   reply("English and French")])
    session = FakeSession({"get_facets": ok('{"values": ["English"]}')})
    answer = await run(llm, session)
    assert answer == "English and French"
    assert session.calls == [("get_facets", {"field": "language"})]
    second = llm.requests[1]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": '{"values": ["English"]}'}


async def test_keeps_full_history_across_multiple_tool_rounds():
    llm = FakeLLM([reply(tool_calls=[tool_call("c1", "resolve_filter_values", {})]),
                   reply(tool_calls=[tool_call("c2", "filter_documents", {})]),
                   reply("done")])
    await run(llm, FakeSession())
    third = llm.requests[2]
    assert third[0]["role"] == "system" and third[1]["role"] == "user"
    assert [m["tool_call_id"] for m in tool_messages(third)] == ["c1", "c2"]


async def test_handles_multiple_tool_calls_in_one_turn():
    llm = FakeLLM([reply(tool_calls=[tool_call("a", "get_facets", {"field": "language"}),
                                     tool_call("b", "get_facets", {"field": "country"})]),
                   reply("ok")])
    session = FakeSession()
    await run(llm, session)
    assert len(session.calls) == 2
    assert [m["tool_call_id"] for m in tool_messages(llm.requests[1])] == ["a", "b"]


async def test_tool_error_is_fed_back_to_llm():
    error = mt.CallToolResult(content=[mt.TextContent(type="text", text="No country named 'Kenia'")],
                              is_error=True)
    llm = FakeLLM([reply(tool_calls=[tool_call("c1", "filter_documents", {"country": "Kenia"})]),
                   reply("fixed")])
    assert await run(llm, FakeSession({"filter_documents": error})) == "fixed"
    assert tool_messages(llm.requests[1])[0]["content"].startswith("ERROR:")


async def test_malformed_tool_arguments_reported_not_raised():
    llm = FakeLLM([reply(tool_calls=[tool_call("c1", "get_facets", "{bad")]), reply("ok")])
    session = FakeSession()
    await run(llm, session)
    assert session.calls == []
    assert "invalid JSON" in tool_messages(llm.requests[1])[0]["content"]


async def test_session_exception_reported_not_raised():
    class Boom(FakeSession):
        async def call_tool(self, name, arguments=None):
            raise RuntimeError("server crashed")

    llm = FakeLLM([reply(tool_calls=[tool_call("c1", "get_facets", {})]), reply("sorry")])
    assert await run(llm, Boom()) == "sorry"
    assert "server crashed" in tool_messages(llm.requests[1])[0]["content"]


async def test_stops_after_max_steps():
    llm = FakeLLM([reply(tool_calls=[tool_call(f"c{i}", "get_facets", {})]) for i in range(3)])
    answer = await run(llm, FakeSession(), max_steps=3)
    assert "Stopped after 3" in answer


async def test_on_event_reports_tool_calls():
    events: list[str] = []
    llm = FakeLLM([reply(tool_calls=[tool_call("c1", "get_facets", {"field": "language"})]),
                   reply("ok")])
    await run(llm, FakeSession(), on_event=events.append)
    assert any("get_facets" in e and "language" in e for e in events)


async def test_dict_arguments_accepted():
    # some OpenAI-compatible servers return arguments already parsed
    call_ = SimpleNamespace(id="c1", type="function",
                            function=SimpleNamespace(name="get_facets", arguments={"field": "language"}))
    session = FakeSession()
    await run(FakeLLM([reply(tool_calls=[call_]), reply("ok")]), session)
    assert session.calls == [("get_facets", {"field": "language"})]


async def test_missing_tool_call_id_is_generated_consistently():
    call_ = tool_call(None, "get_facets", {"field": "language"})
    llm = FakeLLM([reply(tool_calls=[call_]), reply("ok")])
    await run(llm, FakeSession())
    assistant = next(m for m in llm.requests[1] if m.get("tool_calls"))
    generated = assistant["tool_calls"][0]["id"]
    assert generated and tool_messages(llm.requests[1])[0]["tool_call_id"] == generated


async def test_empty_answer_reprompts_once():
    llm = FakeLLM([reply(""), reply("Final answer")])
    assert await run(llm, FakeSession()) == "Final answer"


async def test_empty_answer_twice_gives_clear_message():
    llm = FakeLLM([reply(""), reply("")])
    assert "no answer" in (await run(llm, FakeSession())).lower()


async def test_empty_choices_handled():
    llm = FakeLLM([SimpleNamespace(choices=[]), reply("ok")])
    assert await run(llm, FakeSession()) == "ok"


async def test_system_prompt_all_values_limit_matches_get_facets_max():
    from fastmcp import Client

    from client.agent import SYSTEM_PROMPT
    from server.wb_docs_mcp import mcp
    async with Client(mcp) as client:
        tool = next(t for t in await client.list_tools() if t.name == "get_facets")
    limit_max = tool.inputSchema["properties"]["limit"]["maximum"]
    assert f"limit={limit_max}" in SYSTEM_PROMPT


def test_system_prompt_explain_questions_read_documents_and_cite():
    # "what can you tell me about GST in India" must get an explanation, not just a list of titles
    from client.agent import SYSTEM_PROMPT
    text = SYSTEM_PROMPT.lower()
    assert "what can you tell" in text and "get_document" in text
    assert "opinion" in text  # opinion questions are answered with what the documents say
    assert "most recent" in text  # don't call relevance-ordered results "most recent"


def test_system_prompt_asks_for_plain_terminal_text():
    # answers are printed in a terminal: Markdown (**bold**, tables, #) shows up as clutter
    from client.agent import SYSTEM_PROMPT
    text = SYSTEM_PROMPT.lower()
    assert "plain text" in text and "markdown" in text and "table" in text


def test_tool_result_to_text_caps_length():
    text = tool_result_to_text(ok("x" * (MAX_TOOL_TEXT * 2)))
    assert len(text) <= MAX_TOOL_TEXT + 50 and "truncated" in text


def test_mcp_tools_to_openai_shape():
    tool = mt.Tool(name="get_facets", description="List values",
                   input_schema={"type": "object", "properties": {"field": {"type": "string"}}})
    assert mcp_tools_to_openai([tool]) == [{"type": "function", "function": {
        "name": "get_facets", "description": "List values",
        "parameters": {"type": "object", "properties": {"field": {"type": "string"}}}}}]


def test_load_settings_ollama_default():
    # qwen2.5:3b rebuilt with a 16k context (ollama/Modelfile): the stock 4k window silently
    # drops the system prompt once tool specs + a tool result are in the conversation.
    s = load_settings({})
    assert (s.provider, s.model, s.base_url) == ("ollama", "qwen2.5-16k:3b", "http://localhost:11434/v1")


def test_load_settings_model_override():
    assert load_settings({"LLM_MODEL": "llama3.2:3b"}).model == "llama3.2:3b"


def test_load_settings_groq_requires_key():
    with pytest.raises(SystemExit, match="GROQ_API_KEY"):
        load_settings({"LLM_PROVIDER": "groq"})


def test_main_interactive_exits_cleanly_on_eof(monkeypatch, capsys):
    from client import agent

    def eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    agent.main([])  # must not raise
    assert "World Bank documents assistant" in capsys.readouterr().out


def test_load_settings_groq():
    # llama-3.1-8b-instant was retired by Groq; gpt-oss-20b is free, fast and good at tool calling
    s = load_settings({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "k"})
    assert s.model == "openai/gpt-oss-20b" and "groq.com" in s.base_url and s.api_key == "k"


def _cp1252_stream():
    import io
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def test_main_prints_unicode_answer_to_cp1252_stream(monkeypatch):
    # Windows pipes/redirects default to cp1252; models emit chars like U+2011 that it can't encode
    import sys

    from client import agent

    async def fake_answer(question, settings, **kwargs):
        return "Indonesia‑Country Private Sector Diagnostic"

    monkeypatch.setattr(agent, "answer", fake_answer)
    monkeypatch.setattr("sys.stdout", _cp1252_stream())
    monkeypatch.setattr("sys.stderr", _cp1252_stream())
    agent.main(["q"])
    sys.stdout.flush()
    sys.stderr.flush()
    written = sys.stdout.buffer.getvalue().decode(sys.stdout.encoding)
    assert "Private Sector Diagnostic" in written
    assert b"Error" not in sys.stderr.buffer.getvalue()


def test_main_error_shows_root_cause_of_exception_group(monkeypatch, capsys):
    # anyio wraps failures in an ExceptionGroup whose message hides the real cause
    from client import agent

    async def failing_answer(question, settings, **kwargs):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [ValueError("model `x` does not exist")])

    monkeypatch.setattr(agent, "answer", failing_answer)
    agent.main(["q"])
    err = capsys.readouterr().err
    assert "model `x` does not exist" in err and "ValueError" in err


class RecordingSpinner:
    """Stands in for client.spinner.Spinner and records how main() drives it."""
    log: list = []

    def __init__(self, enabled=None, **kwargs):
        RecordingSpinner.log.append(("enabled", enabled))

    def on_event(self, event):
        RecordingSpinner.log.append(("event", event))

    async def __aenter__(self):
        RecordingSpinner.log.append("enter")
        return self

    async def __aexit__(self, *exc_info):
        RecordingSpinner.log.append("exit")


def _run_main_with_spinner(monkeypatch, argv, isatty=True):
    from client import agent

    seen = {}

    async def fake_answer(question, settings, on_event=None, server_logs=None, **kwargs):
        seen["server_logs"] = server_logs
        on_event("-> get_facets({})")
        return "ok"

    RecordingSpinner.log = []
    monkeypatch.setattr(agent, "Spinner", RecordingSpinner)
    monkeypatch.setattr(agent, "answer", fake_answer)
    monkeypatch.setattr("sys.stderr.isatty", lambda: isatty, raising=False)
    agent.main(argv)
    return RecordingSpinner.log, seen


def test_main_shows_spinner_while_answering(monkeypatch, capsys):
    log, seen = _run_main_with_spinner(monkeypatch, ["q"])
    assert log == [("enabled", True), "enter", ("event", "-> get_facets({})"), "exit"]
    assert seen["server_logs"] is False  # server log lines would scribble over the spinner
    assert capsys.readouterr().out.strip() == "ok"


def test_main_verbose_disables_spinner_and_shows_server_logs(monkeypatch, capsys):
    log, seen = _run_main_with_spinner(monkeypatch, ["-v", "q"])
    assert log[0] == ("enabled", False)
    assert seen["server_logs"] is True
    assert "-> get_facets({})" in capsys.readouterr().err  # verbose trace still printed
