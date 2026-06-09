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
from typing import Optional

from . import _daemon, _priming

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


def _read_stdin_json() -> dict:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


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


def _user_prompt_submit(payload: dict) -> int:
    # Lazy-start the daemon if it's down; never blocks on its ~25s warmup —
    # we use the lexical fallback for this turn and the daemon is warm next time.
    _daemon.ensure_running()
    prompt = payload.get("prompt")
    session_id = payload.get("session_id")
    md = _priming.get_priming(
        prompt if isinstance(prompt, str) else "",
        session_id=session_id if isinstance(session_id, str) else None,
    )
    _emit_additional_context("UserPromptSubmit", md)
    return 0


def _session_start(_payload: dict) -> int:
    # Warm the daemon at session start so the first prompt is already hot.
    _daemon.ensure_running()
    return 0


def dispatch(event: str, payload: Optional[dict] = None) -> int:
    if event not in _EVENT_NAMES:
        print(f"ultan hook: unknown event {event!r}", file=sys.stderr)
        return 2
    data = payload if payload is not None else _read_stdin_json()
    if event == "user-prompt-submit":
        return _user_prompt_submit(data)
    if event == "session-start":
        return _session_start(data)
    # Remaining events (post-tool-use, stop, pre-compact, session-end,
    # pre-tool-use): accepted as no-ops for now — full handlers fold in next.
    return 0
