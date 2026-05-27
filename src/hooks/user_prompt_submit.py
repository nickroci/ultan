"""UserPromptSubmit hook logic — emit turn-start event AND inject context.

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

This module holds the testable logic; ``user-prompt-submit.py`` is the
thin shim Claude Code invokes via settings.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, cast

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _events import HookPayload, append_event  # noqa: E402
from _nudges import render_context, take_nudges  # noqa: E402
from _priming_client import get_priming  # noqa: E402
from scope import current_project_slug  # noqa: E402


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


def _parse_stdin() -> Optional[HookPayload]:
    """Parse the JSON hook_input from stdin; return None on any failure.

    Same Windows-backslash workaround as the other hooks.
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


def _priming_part(prompt: str, session_id: Optional[str]) -> str:
    """Half 2a: ambient priming. Returns rendered markdown or empty string.

    Sub-2 s in the daemon-served path (Unix socket round trip);
    sub-100 ms in the BM25-only fallback. The session_id, when present,
    lets the daemon dedup against entries it has already surfaced to
    this session — without it, dedup is disabled and the agent sees
    full priming as if every turn were a fresh session. The hook
    payload usually carries one; ``get_priming`` tolerates ``None``.
    """
    if not prompt.strip():
        return ""
    try:
        priming_md = get_priming(prompt, session_id=session_id)
    except Exception:
        # Belt and braces: ``get_priming`` is documented as never-raising,
        # but the hook MUST keep running even if something weird happens
        # deeper down.
        return ""
    return priming_md.strip()


def _nudges_part(hook_input: HookPayload) -> str:
    """Half 2b: nudge injection. Returns rendered markdown or empty string.

    Needs a session_id to enforce the per-session budget. If we don't
    have one, skip the nudge half — emitting nudges without bookkeeping
    would let them fire unboundedly.
    """
    session_id = hook_input.get("session_id")
    if not session_id:
        return ""

    # Derive the project slug from the hook's cwd so cross-project
    # nudges (e.g. a vol-predictor nudge queued earlier) don't fire
    # in unrelated repos. ``current_project_slug`` falls back to
    # ``os.getcwd()`` when the hook payload omits cwd.
    hook_cwd = hook_input.get("cwd")
    cwd_arg = hook_cwd if isinstance(hook_cwd, str) else None
    try:
        project_slug: Optional[str] = current_project_slug(cwd_arg)
    except Exception:
        project_slug = None

    try:
        nudges, _consumed = take_nudges(str(session_id), current_project_slug=project_slug)
    except Exception:
        # Best effort. A broken nudges file must never crash the host.
        return ""

    return render_context(nudges) if nudges else ""


def main() -> None:
    # Recursion guard FIRST. Background flush.py invocations don't
    # normally submit user prompts, but the SDK's internal turn
    # machinery does generate prompt-like events in some configurations
    # — guard to be safe.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    hook_input = _parse_stdin()
    if hook_input is None:
        return

    raw_prompt: Any = hook_input.get("prompt", "")
    prompt: str = raw_prompt if isinstance(raw_prompt, str) else ""
    prompt_len = len(prompt)

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
            "text": prompt,
            "prompt_len": prompt_len,
        },
    )

    # Half 2: collect injection parts. Both halves are optional; either,
    # both, or neither may contribute. We assemble first, emit once.
    parts: list[str] = []
    raw_session = hook_input.get("session_id")
    session_id_str: Optional[str] = raw_session if isinstance(raw_session, str) else None
    priming_md = _priming_part(prompt, session_id_str)
    if priming_md:
        parts.append(priming_md)

    nudges_md = _nudges_part(hook_input)
    if nudges_md:
        # Visually separate priming from active nudges so the agent
        # (and the human auditor reading the transcript) can tell
        # which section came from which side of the daemon.
        if parts:
            parts.append("---\n\n## Active nudges\n\n" + nudges_md)
        else:
            parts.append(nudges_md)

    if parts:
        _emit_additional_context("\n\n".join(parts))


if __name__ == "__main__":  # pragma: no cover
    main()
