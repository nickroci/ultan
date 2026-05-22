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


def _dumps(event: dict) -> Optional[bytes]:
    """Compact-ASCII JSON encode; return None on serialisation error."""
    try:
        return json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return None


def _encode_within_budget(event: dict) -> Optional[bytes]:
    """Serialise ``event`` to a JSON line that fits in ``MAX_LINE_BYTES``.

    Cascade:
      1. Try as-is.
      2. If the payload was un-serialisable, drop it and retry.
      3. If the line is over budget, shrink string values in the payload.
      4. If still over budget, drop the payload entirely.

    Returns the encoded bytes (without trailing newline) or None if every
    attempt failed.
    """
    line = _dumps(event)
    if line is None:
        # Un-serialisable payload. Drop and retry — the daemon can still
        # use the event as a turn marker without payload data.
        event["payload"] = {}
        line = _dumps(event)
        if line is None:
            return None

    if len(line) + 1 <= MAX_LINE_BYTES:
        return line

    # Over budget. Shrink string values first.
    if event["payload"]:
        event["payload"] = _truncate_payload(event["payload"], budget=2000)
        line = _dumps(event)
        if line is None:
            return None

    if len(line) + 1 <= MAX_LINE_BYTES:
        return line

    # Still over. Drop the payload entirely.
    event["payload"] = {}
    line = _dumps(event)
    if line is None or len(line) + 1 > MAX_LINE_BYTES:
        return None
    return line


def _write_line(line_bytes: bytes) -> None:
    """Append ``line_bytes + '\\n'`` to the events file. Silently swallows IO."""
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
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
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

    line_bytes = _encode_within_budget(event)
    if line_bytes is None:
        return
    _write_line(line_bytes)
