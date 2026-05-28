"""Shared helpers for the Claude Code hook scripts.

These encode hook-specific contracts that deliberately differ from the
plain ``config``/``scripts`` helpers:

- :func:`ensure_store_dirs` is best-effort and swallows ``OSError`` — a
  hook must never crash the host over a transient mkdir failure, and it
  only needs the handful of dirs it writes to (not the full knowledge
  tree that ``config.ensure_store_dirs`` builds).
- :func:`parse_stdin` tolerates the lone-backslash JSON that Claude Code
  on Windows sometimes emits.

Store-path resolution itself lives in :func:`config.get_config`; hooks
call that for the path, then use these helpers for the hook-flavoured I/O
around it.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional, cast

from _events import HookPayload


def ensure_store_dirs(store: Path) -> None:
    """Best-effort mkdir of the dirs a hook writes to.

    Failures are swallowed: downstream writes already tolerate missing
    parents, and a hook must never crash the host over a transient mkdir
    failure.
    """
    for sub in ("", "state", "knowledge", "daily"):
        try:
            (store / sub if sub else store).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def setup_logging(store: Path, tag: str) -> None:
    """Idempotent logging setup writing to ``<store>/flush.log``.

    ``tag`` labels the emitting hook in each log line (e.g. ``session-end``
    vs ``pre-compact``). Tests can monkeypatch ``logging.basicConfig`` to a
    no-op; a missing/unwritable flush.log must not break the hook.
    """
    try:
        logging.basicConfig(
            filename=str(store / "flush.log"),
            level=logging.INFO,
            format=f"%(asctime)s %(levelname)s [{tag}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    except OSError:
        # Disk full / permission denied / parent missing. Logging is
        # nice-to-have; a missing flush.log must not break the hook.
        pass


def parse_stdin() -> Optional[HookPayload]:
    """Parse the JSON hook payload from stdin; return None on any failure.

    Claude Code on Windows sometimes emits paths with lone backslashes that
    aren't valid JSON escapes; we escape those and retry once before giving
    up.
    """
    try:
        raw = sys.stdin.read()
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            fixed = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw)
            parsed = json.loads(fixed)
    except (json.JSONDecodeError, ValueError, EOFError):
        return None
    if not isinstance(parsed, dict):
        return None
    return cast("HookPayload", parsed)
