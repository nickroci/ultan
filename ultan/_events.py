"""Event capture for the `ultan` Claude Code plugin.

The daemon (`agent-mem-daemon`) learns ONLY from ``events.jsonl``: it tails
that file, seals a "turn" on every ``Stop`` event, and runs the Librarian on
the sealed turn (see ``agent_mem_daemon/buffer.py``). This module is the
plugin-side writer of that stream. The legacy ``src/hooks`` (wired by the
from-source ``/ultan-install`` path) write the same file — exactly one of the
two should be installed per machine, or every event lands twice (see the
README's double-install warning).

It is deliberately stdlib-only and torch-free: it sits on the hook hot path
(every ``PostToolUse``), so ``tests/test_hook_import.py`` guards that nothing
heavy leaks in through here.

Ported from ``src/hooks/_events.py`` (still shipped for the from-source
path): same frozen line schema ``{ts, session_id, type, cwd, payload}``, same
``O_APPEND`` atomic write, same 3 KiB budget, same ``CLAUDE_INVOKED_BY``
recursion guard. The only change is the path resolves via
:func:`ultan._daemon._home`, so this module carries no dependency on the
heavy ``src/scripts/config``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from ._daemon import _home  # pyright: ignore[reportPrivateUsage]  # intra-package

# Keep event lines small. Atomicity rests on single ``os.write`` calls with
# ``O_APPEND`` to a regular file — which events.jsonl is. (It would NOT hold
# for a FIFO on macOS, where PIPE_BUF is 512 B < 3 KiB; don't point this at a
# pipe.) 3 KiB matches the legacy budget; the daemon never needs more — the
# full tool input/output stays in the transcript, which the Librarian reads.
MAX_LINE_BYTES = 3072


def _events_path() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}/events.jsonl``."""
    return _home() / "events.jsonl"


def _truncate_payload(payload: Mapping[str, Any], budget: int) -> dict[str, Any]:
    """Best-effort shrink of each string value to ``budget`` UTF-8 BYTES.

    Bytes, not characters: the line limit is bytes and ``ensure_ascii=False``
    means non-ASCII text inflates several-fold on encode. Only strings are
    truncated; other types pass through. Not guaranteed to fit — the final
    length check happens at the line level in :func:`_encode_within_budget`.
    """
    out: dict[str, Any] = {}
    for k, v in payload.items():
        value: Any = v
        if isinstance(value, str) and len(value.encode("utf-8")) > budget:
            clipped = value.encode("utf-8")[: max(0, budget - 3)]
            # errors="ignore" drops a multi-byte char cut in half at the edge.
            out[k] = clipped.decode("utf-8", errors="ignore") + "..."
        elif isinstance(value, dict):
            nested = cast("Mapping[str, Any]", value)
            out[k] = _truncate_payload(nested, budget=budget)
        else:
            out[k] = value
    return out


def _dumps(event: Mapping[str, Any]) -> Optional[bytes]:
    """Compact-ASCII JSON encode; return None on serialisation error."""
    try:
        return json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return None


def _encode_within_budget(event: dict[str, Any]) -> Optional[bytes]:
    """Serialise ``event`` to a JSON line that fits in ``MAX_LINE_BYTES``.

    Cascade: try as-is → drop an un-serialisable payload → shrink string
    values → drop the payload entirely. Returns the encoded bytes (no
    trailing newline) or None if every attempt failed.
    """
    empty_payload: dict[str, Any] = {}
    line = _dumps(event)
    if line is None:
        # Un-serialisable payload. Drop and retry — the daemon can still use
        # the event as a turn marker without payload data.
        event["payload"] = empty_payload
        line = _dumps(event)
        if line is None:
            return None

    if len(line) + 1 <= MAX_LINE_BYTES:
        return line

    # Over budget. Shrink string values at progressively tighter per-string
    # byte budgets: a payload can carry several large strings at once
    # (PostToolUse ``content`` + ``response``), so one fixed cut that fits a
    # single string can still overflow the line. Tighten until it fits.
    payload_obj: Any = event["payload"]
    if payload_obj:
        original_payload = cast("Mapping[str, Any]", payload_obj)
        for budget in (2000, 900, 400, 150):
            event["payload"] = _truncate_payload(original_payload, budget=budget)
            line = _dumps(event)
            if line is None:
                return None
            if len(line) + 1 <= MAX_LINE_BYTES:
                return line

    # Still over. Drop the payload entirely.
    event["payload"] = empty_payload
    line = _dumps(event)
    if line is None or len(line) + 1 > MAX_LINE_BYTES:
        return None
    return line


def _write_line(line_bytes: bytes) -> None:
    """Append ``line_bytes + '\\n'`` to the events file. Silently swallows IO."""
    try:
        path = _events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    # O_APPEND for atomicity across concurrent writers (parallel Claude Code
    # sessions). Open low-level, single write(), close — no buffering layer.
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
        event_type: ``PostToolUse`` | ``Stop`` | ``SessionEnd`` |
            ``UserPromptSubmit`` — the types the daemon's tailer accepts
            (``agent_mem_daemon/ingest.py``). ``Stop`` and ``SessionEnd``
            are what seal turns and drive the Librarian, so they matter even
            with an empty payload.
        hook_input: the parsed stdin dict from Claude Code. We read
            ``session_id`` (required by the daemon) and ``cwd`` — the stdin
            field, NOT ``os.getcwd()``, because the hook process can run under
            a different cwd than the active session.
        payload: optional small dict; ``None`` is normalised to ``{}``.

    Side effects: appends one ``\\n``-terminated JSON line to
    ``${AGENT_MEM_HOME:-~/.agent-mem}/events.jsonl``. Every failure (disk full,
    permission denied, malformed input) is swallowed — a broken events file
    must never crash the host agent; the daemon simply sees a gap.
    """
    # Recursion guard — must come first. The daemon spawns the Librarian/Scholar
    # as Agent-SDK subprocesses with CLAUDE_INVOKED_BY set (see
    # agent_mem_daemon/llm.py); without this their own Read/Edit/Write calls
    # would write ghost events and feed a capture loop. (They also run with
    # setting_sources=[] so the plugin hooks shouldn't load at all — this is the
    # belt to that suspenders.)
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    # The daemon rejects events with no session_id, so drop rather than emit a
    # malformed line (better a gap than corruption).
    session_id = hook_input.get("session_id")
    if not session_id:
        return

    # cwd MUST come from the stdin field, never os.getcwd().
    cwd_raw: Any = hook_input.get("cwd")
    cwd = cwd_raw if isinstance(cwd_raw, str) else None

    event: dict[str, Any] = {
        "ts": time.time(),
        "session_id": str(session_id),
        "type": event_type,
        "cwd": cwd,
        "payload": dict(payload) if payload else {},
    }

    line = _encode_within_budget(event)
    if line is not None:
        _write_line(line)


# ── Per-event payload builders ──────────────────────────────────────────────
#
# Field names align with what the daemon's Librarian extracts (see
# ``agent_mem_daemon/librarian_prompt`` ``_TEXT_KEYS``): ``content`` carries the
# concrete code/command/prompt; ``summary`` a one-line synthesized fallback.


def _first_str_field(d: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty string field from ``d`` keyed by ``keys``."""
    for key in keys:
        v: Any = d.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _response_text(tool_response: object) -> str:
    """Flatten the ``tool_response`` field to a plain string."""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        resp = cast("Mapping[str, Any]", tool_response)
        return str(resp.get("content") or resp.get("output") or "")
    return ""


def post_tool_use_payload(hook_input: Mapping[str, Any]) -> dict[str, Any]:
    """Small fixed-shape payload for a ``PostToolUse`` event.

    Frozen shape: ``{role, tool, ok, summary, content, response}``. The full
    tool input/output is intentionally NOT embedded — it's in the transcript
    and would blow the per-line budget on big Read/Bash payloads.
    """
    raw_name: Any = hook_input.get("tool_name") or "unknown"
    name = raw_name if isinstance(raw_name, str) else str(raw_name)
    # PostToolUse carries ``tool_response``; the failure variant carries
    # ``error``. Default to True so an odd shape isn't falsely flagged.
    ok = not hook_input.get("error")

    raw_tool_input: Any = hook_input.get("tool_input") or {}
    target = ""
    content = ""
    if isinstance(raw_tool_input, dict):
        tool_input = cast("Mapping[str, Any]", raw_tool_input)
        target = _first_str_field(tool_input, ("file_path", "path", "url", "pattern", "query"))
        content = _first_str_field(tool_input, ("new_string", "content", "command", "code"))

    bits = [name]
    if target:
        bits.append(f"on {target}")
    bits.append("ok" if ok else "FAILED")

    return {
        "role": "assistant",
        "tool": name,
        "ok": ok,
        "summary": " ".join(bits),
        "content": content,
        "response": _response_text(hook_input.get("tool_response")),
    }


def user_prompt_payload(hook_input: Mapping[str, Any]) -> dict[str, Any]:
    """Small payload for a ``UserPromptSubmit`` event — carries the prompt as
    ``content`` so the Librarian sees the turn's intent. Oversize prompts are
    truncated to the line budget by :func:`_encode_within_budget`."""
    raw_prompt: Any = hook_input.get("prompt")
    prompt = raw_prompt if isinstance(raw_prompt, str) else ""
    # Key is ``content`` (legacy src/hooks writes ``text``) — the daemon's
    # librarian_prompt._TEXT_KEYS accepts both, so the divergence is benign.
    return {"role": "user", "content": prompt}
