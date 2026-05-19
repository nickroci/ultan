"""UserPromptSubmit hook — emit turn-start event AND inject pending nudges.

Two responsibilities, both kept fast (< 50 ms target — pure file I/O,
zero LLM calls):

1. **Daemon event.** Append a turn-start marker to the JSONL stream the
   daemon tails. Frozen schema, same as the rest of the hook layer.
   Payload: ``{"prompt_len": <int>}``. The prompt text itself is never
   embedded (may contain secrets; not needed for scheduling).

2. **Nudge injection** (PLAN §5 / Phase 3). The Scholar has written
   ratified interrupts to ``~/.agent-mem/pending-nudges.md``. We read,
   clear, enforce the 1/turn + 3/session budget, and emit the survivors
   as ``additionalContext`` via the hook's stdout JSON.

The two halves are independent: a missing/empty nudges file just means
no context gets injected. A missing session_id means we can't enforce
the per-session budget, so we skip the nudge half entirely.

Recursion guard via ``CLAUDE_INVOKED_BY`` runs first — the Scholar's
SDK subprocess inside the daemon can submit prompt-like events, and
without the guard we'd inject nudges back into the Scholar's own
context. That would be unhelpful at best, infinite-loop-y at worst.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Recursion guard FIRST. Background flush.py invocations don't normally
# submit user prompts, but the SDK's internal turn machinery does
# generate prompt-like events in some configurations — guard to be safe.
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
from _nudges import render_context, take_nudges  # noqa: E402


def _emit_additional_context(context: str) -> None:
    """Print the hook's stdout JSON exactly once.

    Claude Code expects a single JSON object on stdout (or nothing).
    The schema follows session-start.py's pattern.
    """
    if not context:
        return
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    print(json.dumps(output))


def main() -> None:
    try:
        raw_input = sys.stdin.read()
        try:
            hook_input: dict = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw_input)
            hook_input = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError):
        return

    if not isinstance(hook_input, dict):
        return

    prompt = hook_input.get("prompt", "")
    prompt_len = len(prompt) if isinstance(prompt, str) else 0

    # Half 1: daemon event. Append before doing the nudge work — the
    # daemon's view of "turn started" should not be gated on whether we
    # have nudges to inject. Include the actual prompt text — the
    # Librarian needs content to extract candidate lessons. Long prompts
    # are truncated by _events.append_event's cascade.
    append_event(
        "UserPromptSubmit",
        hook_input,
        payload={
            "role": "user",
            "text": prompt if isinstance(prompt, str) else "",
            "prompt_len": prompt_len,
        },
    )

    # Half 2: nudge injection. Needs a session_id to enforce the
    # per-session budget. If we don't have one, skip — emitting nudges
    # without bookkeeping would let them fire unboundedly.
    session_id = hook_input.get("session_id")
    if not session_id:
        return

    try:
        nudges, _consumed = take_nudges(str(session_id))
    except Exception:
        # Best effort. A broken nudges file must never crash the host.
        return

    if nudges:
        _emit_additional_context(render_context(nudges))


if __name__ == "__main__":
    main()
