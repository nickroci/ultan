"""PostToolUse hook — append a turn-event to the daemon's input stream.

Fires after every tool call. We emit one tiny event line and exit. The
full tool input/output is intentionally NOT included; it's already in
the transcript and would blow the per-line PIPE_BUF budget on big Read
or Bash payloads. The daemon's Librarian reads the transcript when it
needs more context.

Payload shape (frozen):
    {"tool_name": "<name>", "ok": true|false}

``ok`` is best-effort: Claude Code's PostToolUse hook fires for both
success and failure (there's also a separate PostToolUseFailure event,
but configuring both would mean the daemon sees every failure twice).
We infer success from presence of ``tool_response`` and absence of
``error``.

Latency target: < 5 ms. No SDK calls, no subprocesses, no logging.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Recursion guard FIRST — before any imports or work. If we're inside
# an SDK call spawned by flush.py, emitting a PostToolUse event for
# every Read/Edit the SDK does would flood the daemon with ghost events.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _events import append_event  # noqa: E402


def main() -> None:
    # Read + parse stdin. Same Windows-backslash workaround as the other
    # hooks — Claude Code on Windows sometimes emits paths with lone
    # backslashes that aren't valid JSON escapes.
    try:
        raw_input = sys.stdin.read()
        try:
            hook_input: dict = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw_input)
            hook_input = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError):
        # Malformed input — drop the event silently. Daemon will see a
        # gap but nothing breaks.
        return

    if not isinstance(hook_input, dict):
        return

    tool_name = hook_input.get("tool_name")
    # Success heuristic: PostToolUse carries ``tool_response``; the
    # corresponding failure event (PostToolUseFailure) carries ``error``
    # instead. If neither is present (unexpected shape), default to
    # ``True`` so we don't falsely flag every event.
    if "error" in hook_input and hook_input["error"]:
        ok = False
    else:
        ok = True

    # Include a brief summary of the call. Long values are truncated by
    # _events.append_event's cascade. Field names align with what the
    # daemon's Librarian extracts from payloads (see
    # ``librarian_prompt._TEXT_KEYS``): ``content`` gets the actual code
    # / command / new_string when present; ``summary`` gets a one-line
    # synthesized description as a fallback.
    tool_input = hook_input.get("tool_input") or {}
    tool_response = hook_input.get("tool_response")
    name = str(tool_name) if tool_name else "unknown"

    target = ""
    content = ""
    if isinstance(tool_input, dict):
        for key in ("file_path", "path", "url", "pattern", "query"):
            v = tool_input.get(key)
            if isinstance(v, str) and v:
                target = v
                break
        for key in ("new_string", "content", "command", "code"):
            v = tool_input.get(key)
            if isinstance(v, str) and v:
                content = v
                break

    response_text = ""
    if isinstance(tool_response, str):
        response_text = tool_response
    elif isinstance(tool_response, dict):
        response_text = str(tool_response.get("content") or tool_response.get("output") or "")

    bits = [name]
    if target:
        bits.append(f"on {target}")
    bits.append("ok" if ok else "FAILED")
    summary = " ".join(bits)

    payload = {
        "role": "assistant",
        "tool": name,
        "ok": ok,
        "summary": summary,
        "content": content,
        "response": response_text,
    }
    append_event("PostToolUse", hook_input, payload=payload)


if __name__ == "__main__":
    main()
