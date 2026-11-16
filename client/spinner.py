"""A one-line terminal loading indicator: `| Calling filter_documents... 4s`.

Written to stderr and only on a real terminal, so piped or redirected answers stay clean.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from typing import TextIO

FRAMES = "|/-\\"  # plain ASCII renders in every Windows console
MAX_LINE = 100  # tool-call lines are shortened to fit a normal terminal width


class Spinner:
    def __init__(self, stream: TextIO | None = None, enabled: bool | None = None,
                 interval: float = 0.1) -> None:
        self.stream = stream or sys.stderr
        self.enabled = _is_tty(self.stream) if enabled is None else enabled
        self.interval = interval
        self.label = "Thinking"
        self._task: asyncio.Task | None = None
        self._width = 0

    def on_event(self, event: str) -> None:
        """Follow the agent loop's events: '-> tool(...)' starts a call, '<- ...' is its result.

        Each tool call is also printed as its own lasting line above the spinner, because a
        World Bank call finishes in a fraction of a second and the label alone flashes by.
        """
        if event.startswith("-> "):
            name, _, args = event[3:].partition("(")
            self.label = "Calling " + name
            if self.enabled:
                line = f"  > {name} {args.removesuffix(')')}".rstrip()
                if len(line) > MAX_LINE:
                    line = line[:MAX_LINE - 3] + "..."
                self._write("\r" + " " * self._width + "\r" + line + "\n")
        elif event.startswith("<- "):
            self.label = "Thinking"

    async def __aenter__(self) -> "Spinner":
        if self.enabled:
            self._task = asyncio.create_task(self._spin(time.monotonic()))
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._write("\r" + " " * self._width + "\r")

    async def _spin(self, started: float) -> None:
        frame = 0
        while True:
            text = f"{FRAMES[frame % len(FRAMES)]} {self.label}... {time.monotonic() - started:.0f}s"
            self._width = max(self._width, len(text))
            self._write("\r" + text.ljust(self._width))
            frame += 1
            await asyncio.sleep(self.interval)

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()


def _is_tty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False
