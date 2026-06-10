"""Light hook handlers for the thin `ultan` wrapper.

Invoked as `ultan hook <event>` from Claude Code's settings.json. The hot path
(user-prompt-submit) runs as a fresh process every turn under a ~2s budget, so
it MUST stay fast and torch-free: it talks to the daemon over a Unix socket,
lazy-starts the daemon if it's down, and falls back to a crude stdlib lexical
scan. NO heavy/ML imports belong in this module or its transitive deps —
tests/test_hook_import.py guards that invariant.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional, cast

from . import _daemon, _events, _priming

# Hook events we accept (kebab-case as written into settings.json). The
# hookEventName Claude Code expects back is the CamelCase form.
_EVENT_NAMES = {
    "session-start": "SessionStart",
    "user-prompt-submit": "UserPromptSubmit",
    "post-tool-use": "PostToolUse",
    "stop": "Stop",
    "pre-compact": "PreCompact",
    "session-end": "SessionEnd",
    "pre-tool-use": "PreToolUse",
}


def _read_stdin_json() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    # ValueError covers UnicodeDecodeError: under a UTF-8 locale, non-UTF-8
    # stdin raises at read() and must degrade to {}, not a traceback.
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return cast("dict[str, Any]", data) if isinstance(data, dict) else {}


def _emit_additional_context(event_name: str, context: str) -> None:
    if not context:
        return
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": event_name,
                    "additionalContext": context,
                }
            }
        )
    )


def _user_prompt_submit(payload: dict[str, Any]) -> int:
    # Lazy-start the daemon if it's down; never blocks on its ~25s warmup —
    # we use the lexical fallback for this turn and the daemon is warm next time.
    _daemon.ensure_running()
    # Capture: record the prompt so the daemon's Librarian sees the turn's
    # intent. Independent of the stdout priming below (file write, not stdout).
    _events.append_event("UserPromptSubmit", payload, payload=_events.user_prompt_payload(payload))
    prompt = payload.get("prompt")
    session_id = payload.get("session_id")
    md = _priming.get_priming(
        prompt if isinstance(prompt, str) else "",
        session_id=session_id if isinstance(session_id, str) else None,
    )
    # Fallback honesty: while the daemon warms (first start loads models for
    # minutes), priming comes from the crude lexical scan. Say so, so the
    # agent treats the bullets as provisional rather than the library's best.
    if md and _daemon.status() == "warming":
        md += (
            "\n*Ultan's daemon is still warming up — the bullets above are "
            "lexical-fallback results; full ranked recall returns within a "
            "minute or two.*\n"
        )
    _emit_additional_context("UserPromptSubmit", md)
    return 0


def _session_start(payload: dict[str, Any]) -> int:
    # Warm the daemon at session start so the first prompt is already hot.
    _daemon.ensure_running()
    # Capture the session boundary (legacy parity: src/hooks/session_start.py
    # logged {"source": startup|resume|clear|compact}). No-ops harmlessly when
    # stdin was empty — no session_id means append_event drops the line.
    source = payload.get("source")
    _events.append_event(
        "SessionStart",
        payload,
        payload={"source": source if isinstance(source, str) else "unknown"},
    )
    return 0


def dispatch(event: str, payload: Optional[dict[str, Any]] = None) -> int:
    if event not in _EVENT_NAMES:
        # Exit 1, NOT 2: in the Claude Code hook protocol exit 2 is the
        # blocking signal (denies tools / erases prompts). A misconfigured
        # event name must fail soft.
        print(f"ultan hook: unknown event {event!r}", file=sys.stderr)
        return 1
    data = payload if payload is not None else _read_stdin_json()
    if event == "user-prompt-submit":
        return _user_prompt_submit(data)
    if event == "session-start":
        return _session_start(data)
    # Capture path: feed the daemon's event stream. PostToolUse accumulates
    # into the open turn; Stop seals it (→ Librarian runs); SessionEnd seals
    # the final turn and marks the session over (see agent_mem_daemon/buffer.py).
    if event == "post-tool-use":
        _events.append_event("PostToolUse", data, payload=_events.post_tool_use_payload(data))
        return 0
    if event == "stop":
        _events.append_event("Stop", data)
        return 0
    if event == "session-end":
        _events.append_event("SessionEnd", data)
        return 0
    if event == "pre-compact":
        # Legacy parity (src/hooks/pre_compact.py): emit a SessionEnd to force
        # a turn seal + Librarian pass BEFORE compaction discards transcript
        # detail. The type is SessionEnd on purpose — SessionEnd is the
        # daemon's turn-sealing path; it has no PreCompact handling.
        _events.append_event("SessionEnd", data, payload={"source": "pre-compact"})
        return 0
    # pre-tool-use: accepted no-op (PostToolUse already captures tool usage).
    # Kept in _EVENT_NAMES so a stale hooks.json wiring fails soft.
    return 0
