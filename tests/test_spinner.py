"""Tests for the terminal loading indicator."""

from __future__ import annotations

import asyncio
import io

from client.spinner import Spinner


async def test_spinner_shows_label_and_elapsed_then_clears():
    out = io.StringIO()
    async with Spinner(stream=out, enabled=True, interval=0.01):
        await asyncio.sleep(0.05)
    text = out.getvalue()
    assert "Thinking..." in text and "0s" in text
    # last write blanks the line and returns the cursor, so the answer prints on a clean line
    assert text.endswith("\r") and text.rstrip("\r").split("\r")[-1].strip() == ""


async def test_spinner_disabled_writes_nothing():
    out = io.StringIO()
    async with Spinner(stream=out, enabled=False, interval=0.01):
        await asyncio.sleep(0.03)
    assert out.getvalue() == ""


def test_spinner_label_follows_tool_events():
    spinner = Spinner(stream=io.StringIO(), enabled=True)
    spinner.on_event('-> filter_documents({"country": "Brazil"})')
    assert spinner.label == "Calling filter_documents"
    spinner.on_event('<- {"total": 3}')
    assert spinner.label == "Thinking"
    spinner.on_event("[groq:openai/gpt-oss-20b] tools: get_facets")  # other events: unchanged
    assert spinner.label == "Thinking"


async def test_tool_calls_stay_on_screen_as_their_own_lines():
    # the label alone flashes by (a World Bank call takes ~0.3 s); each call gets a lasting line
    out = io.StringIO()
    async with Spinner(stream=out, enabled=True, interval=0.01) as spinner:
        await asyncio.sleep(0.02)
        spinner.on_event('-> search_documents({"query":"GST India","rows":5})')
        await asyncio.sleep(0.02)
    lines = out.getvalue().split("\n")
    assert any(line.strip().endswith('> search_documents {"query":"GST India","rows":5}') for line in lines)


def test_long_tool_arguments_are_shortened():
    out = io.StringIO()
    spinner = Spinner(stream=out, enabled=True)
    spinner.on_event("-> filter_documents(" + '{"query":"' + "x" * 300 + '"})')
    line = out.getvalue().strip("\r\n ").split("\r")[-1]
    assert len(line) <= 100 and line.endswith("...")


def test_disabled_spinner_prints_no_tool_lines():
    out = io.StringIO()
    Spinner(stream=out, enabled=False).on_event('-> get_facets({"field":"language"})')
    assert out.getvalue() == ""


async def test_spinner_clears_even_when_body_raises():
    out = io.StringIO()
    try:
        async with Spinner(stream=out, enabled=True, interval=0.01):
            await asyncio.sleep(0.02)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert out.getvalue().endswith("\r")


def test_spinner_enabled_by_default_only_on_a_terminal():
    assert Spinner(stream=io.StringIO()).enabled is False  # StringIO is not a TTY
