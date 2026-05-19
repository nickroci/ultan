"""Shared helper for appending events to ``~/.agent-mem/events.jsonl``.

Every hook calls :func:`append_event` exactly once. The function:

1. Honours the ``CLAUDE_INVOKED_BY`` recursion guard — if set, we are
   inside an Agent-SDK subprocess spawned by another hook (flush.py),
   and we must not emit ghost events back to the daemon.
2. Resolves the events file path under ``${AGENT_MEM_HOME:-~/.agent-mem}``,
   reusing :mod:`config` so the env-var override stays in one place.
3. Builds the frozen event-line schema (PLAN §7 item 4):
   ``{"ts","session_id","type","cwd","payload"}``.
4. Writes the JSON line with ``O_APPEND`` so concurrent writes from
   parallel Claude Code sessions stay atomic for lines under PIPE_BUF
   (~4 KiB on Linux/macOS).
5. Silently swallows any I/O failure — hooks must never crash the host.

The line is *never* longer than the PIPE_BUF limit because:

- ``payload`` is constrained at the call-site to small, fixed-shape
  dicts (tool_name, ok, prompt_len, source). No tool inputs / outputs /
  prompt text are ever embedded.
- A defensive truncation pass at the end caps any pathological line at
  ``MAX_LINE_BYTES`` (3 KiB) before write.

Latency target: well under 5 ms in the common case. No subprocesses, no
LLM calls, no locks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

# Make scripts/ importable so we can reuse the AGENT_MEM_HOME helper that
# already lives in config.py. We import lazily inside _events_path() so a
# config import failure (unlikely but possible) doesn't break the recursion-
# guard short-circuit.
_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# PIPE_BUF is 512 bytes on POSIX minimum, 4096 on Linux, 512 on macOS for
# pipes — but for regular files O_APPEND is atomic per-write at any size
# on Linux, and macOS guarantees atomicity up to PIPE_BUF for pipes only.
# To stay safe across platforms (and across the unlikely future case where
# someone points events.jsonl at a FIFO), keep lines under 3 KiB.
MAX_LINE_BYTES = 3072


def _events_path() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}/events.jsonl``.

    Reuses ``config.STORE_DIR`` so the env-var override logic stays in one
    place. Returns the path even if the parent dir doesn't exist yet —
    caller creates it on first write.
    """
    from config import STORE_DIR  # noqa: E402  — see module docstring

    return STORE_DIR / "events.jsonl"


def _truncate_payload(payload: Mapping[str, Any], budget: int) -> dict:
    """Best-effort shrink of payload values to fit in ``budget`` bytes.

    Only strings are truncated; other types pass through. The result is
    not guaranteed to fit — if every value is already short, the caller
    just gets back roughly what they passed in. The final length check
    happens at the line level in :func:`append_event`.
    """
    out: dict = {}
    for k, v in payload.items():
        if isinstance(v, str) and len(v) > budget:
            out[k] = v[: max(0, budget - 3)] + "..."
        elif isinstance(v, dict):
            out[k] = _truncate_payload(v, budget=budget)
        else:
            out[k] = v
    return out


def append_event(
    event_type: str,
    hook_input: Mapping[str, Any],
    payload: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append one JSONL event line to the daemon's input file.

    Args:
        event_type: One of ``PostToolUse``, ``Stop``, ``SessionEnd``,
            ``UserPromptSubmit``, ``SessionStart``. Other strings are
            accepted (daemon logs unknown types at DEBUG) but the agreed
            five cover the current contract.
        hook_input: The parsed stdin dict from Claude Code. We read
            ``session_id`` and ``cwd`` (the stdin field — NOT
            ``os.getcwd()`` — because the hook process may run under a
            different cwd than the active session).
        payload: Optional small dict. Keep it tiny; the full content
            stays in the transcript. None is normalised to ``{}``.

    Side effects: appends one ``\\n``-terminated JSON line to
    ``${AGENT_MEM_HOME:-~/.agent-mem}/events.jsonl``. Failures (disk full,
    permission denied, etc.) are silently swallowed — a broken events
    file must never crash the host agent. Errors aren't logged to stderr
    either, because Claude Code captures hook stderr and surfacing
    write-errors mid-turn would be more annoying than useful; the daemon
    will simply see a gap in its event stream.
    """
    # Recursion guard. Must come before any work — see flush.py and
    # session-end.py for the matching guard at the top of those files.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    # Required fields. The daemon rejects events missing session_id, so
    # if we can't get one we drop the event rather than emit a malformed
    # line. (Better a gap than corruption.)
    session_id = hook_input.get("session_id")
    if not session_id:
        return

    # cwd MUST come from the stdin field, never os.getcwd() — see PLAN
    # decision pinned by Agent 1.
    cwd = hook_input.get("cwd")

    event: dict = {
        "ts": time.time(),  # unix-float; the daemon also accepts ISO-8601.
        "session_id": str(session_id),
        "type": event_type,
        "cwd": cwd if isinstance(cwd, str) else None,
        "payload": dict(payload) if payload else {},
    }

    # Serialise. ``separators`` keeps the line compact; ``ensure_ascii``
    # avoids surprising the daemon's parser with non-ASCII (it handles
    # it fine, but compact ASCII is one fewer thing to worry about over
    # the wire).
    try:
        line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        # Payload contained something un-serialisable. Drop the payload,
        # keep the event — the daemon can still use it as a turn marker.
        event["payload"] = {}
        try:
            line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return

    line_bytes = line.encode("utf-8")
    if len(line_bytes) + 1 > MAX_LINE_BYTES:
        # Over budget. Try to shrink the payload first.
        if event["payload"]:
            shrunk = _truncate_payload(event["payload"], budget=2000)
            event["payload"] = shrunk
            try:
                line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
                line_bytes = line.encode("utf-8")
            except (TypeError, ValueError):
                return
        # Still too big? Drop the payload entirely.
        if len(line_bytes) + 1 > MAX_LINE_BYTES:
            event["payload"] = {}
            try:
                line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
                line_bytes = line.encode("utf-8")
            except (TypeError, ValueError):
                return

    # Resolve path + ensure parent dir exists. The parent-dir create is
    # idempotent and cheap (<1 ms) but only happens once per process
    # because the OS caches the inode.
    try:
        path = _events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ImportError):
        return

    # O_APPEND for atomicity. We open low-level, write, close — no
    # buffering layer to flush. ``os.O_APPEND`` plus a single write()
    # under PIPE_BUF is atomic across concurrent writers on Linux+macOS
    # for regular files; see open(2) and write(2).
    try:
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
    except OSError:
        return

    try:
        os.write(fd, line_bytes + b"\n")
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
