"""UserPromptSubmit hook — emit turn-start event AND inject context.

Three responsibilities, all kept fast (< 200 ms target end-to-end —
file I/O + one Unix-socket round trip to the daemon, zero LLM calls):

1. **Daemon event.** Append a turn-start marker to the JSONL stream the
   daemon tails. Frozen schema, same as the rest of the hook layer.
   Payload: ``{"prompt_len": <int>}``. The prompt text itself is never
   embedded (may contain secrets; not needed for scheduling).

2. **Ambient priming (Tier 1) — daemon RPC.** We ask the daemon (over
   its ``~/.agent-mem/priming.sock`` Unix socket) for a rendered
   priming snippet keyed on the user's *current prompt*. The daemon
   has the embedding model warm in memory and returns rendered
   markdown in <100 ms. If the daemon is down or the socket is
   missing, ``_priming_client.get_priming`` transparently falls back
   to an in-hook lexical scan of the knowledge tree so the agent
   still sees *something* — never nothing.

3. **Nudge injection** (PLAN §5 / Phase 3). The Scholar has written
   ratified interrupts to ``~/.agent-mem/pending-nudges.md``. We read,
   clear, enforce the 1/turn + 3/session budget, and emit the survivors
   as ``additionalContext`` via the hook's stdout JSON.

All three halves are independent: a missing/empty file at any step just
means that section is omitted from the injected context. A missing
session_id means we can't enforce the per-session budget, so we skip
the nudge half (priming has no per-session bookkeeping and still runs).

Recursion guard via ``CLAUDE_INVOKED_BY`` runs first — the Scholar's
SDK subprocess inside the daemon can submit prompt-like events, and
without the guard we'd inject context back into the Scholar's own
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
from _priming_client import get_priming  # noqa: E402


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

    # Half 2: collect injection parts. Both halves are optional; either,
    # both, or neither may contribute. We assemble first, emit once.
    parts: list[str] = []

    # Half 2a: ambient priming (Tier 1). Sub-200 ms in the daemon-served
    # path (Unix socket round trip); sub-100 ms in the BM25-only
    # fallback. Runs even when session_id is missing — priming is
    # session-agnostic and ``get_priming`` enforces its own char budget.
    if isinstance(prompt, str) and prompt.strip():
        try:
            priming_md = get_priming(prompt)
        except Exception:
            # Belt and braces: ``get_priming`` is documented as
            # never-raising, but the hook MUST keep running even if
            # something weird happens deeper down.
            priming_md = ""
        if priming_md.strip():
            parts.append(priming_md.strip())

    # Half 2b: nudge injection. Needs a session_id to enforce the
    # per-session budget. If we don't have one, skip the nudge half —
    # emitting nudges without bookkeeping would let them fire unboundedly.
    # Priming may still have been added above.
    session_id = hook_input.get("session_id")
    if session_id:
        try:
            nudges, _consumed = take_nudges(str(session_id))
        except Exception:
            # Best effort. A broken nudges file must never crash the host.
            nudges = []

        if nudges:
            rendered = render_context(nudges)
            if rendered:
                # Visually separate priming from active nudges so the agent
                # (and the human auditor reading the transcript) can tell
                # which section came from which side of the daemon.
                if parts:
                    parts.append("---\n\n## Active nudges\n\n" + rendered)
                else:
                    parts.append(rendered)

    if parts:
        _emit_additional_context("\n\n".join(parts))


if __name__ == "__main__":
    main()
