"""PreToolUse hook logic — synchronous deterministic blocker check + event stream.

This is **Tier 3** of the retrieval pipeline (see ``README.md``). The
nudge pipeline that runs through ``user-prompt-submit.py`` is post-hoc
and asynchronous — by the time a "never deploy to prod without
approval" nudge lands, the destructive ``gcloud deploy`` has already
run. PreToolUse is the only synchronous interrupt point Claude Code
exposes: it fires **before** the tool executes and a hook response of
``permissionDecision: deny`` blocks the call.

Responsibilities, in order:

1. Recursion guard. If ``CLAUDE_INVOKED_BY`` is set we are inside an
   Agent-SDK subprocess spawned by another hook (the daemon's Scholar,
   primarily). We exit immediately so:
   - we never block the Scholar's own Edit/Write calls (which would
     deadlock the daemon),
   - we never spam the event stream with ghost PreToolUse events for
     every Read/Glob the Scholar does.
2. Read the tool name + input from stdin.
3. Load blockers (cached by knowledge-dir sentinel mtime — see
   ``_blockers.py``) and ask :func:`find_match` whether this call
   matches any ``severity: block`` rule.
4. If matched → emit a ``deny`` response on stdout and return. Claude
   Code surfaces ``permissionDecisionReason`` to the agent so it can
   either rephrase or back off and ask the user.
5. **Whether or not we denied**, append the PreToolUse event to the
   daemon's stream so the daemon-side modules (Librarian, Scholar) can
   observe both the call and the block decision in real time.

Latency budget: < 100 ms. This hook runs synchronously in front of
every tool call so every millisecond is felt. No LLM calls, no
subprocesses, no Read/Write of unrelated files.

This module holds the testable logic; ``pre-tool-use.py`` is the thin
shim Claude Code invokes via settings.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, cast

from _blockers import find_match, load_blockers, rel_to_knowledge
from _events import EventPayload, HookPayload, append_event
from config import get_config


def _read_hook_input() -> HookPayload:
    """Parse the stdin JSON Claude Code sends, with the same Windows
    backslash workaround used by the other hooks.
    """
    raw = sys.stdin.read()
    parsed: Any
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Lone backslashes in file paths sometimes break strict JSON
        # parsing on Windows; double them up and retry.
        fixed = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw)
        parsed = json.loads(fixed)
    if not isinstance(parsed, dict):
        # Caller catches and short-circuits.
        raise ValueError("hook payload was not a JSON object")
    return cast("HookPayload", parsed)


def _emit_hook_output(extra: dict[str, Any]) -> None:
    """Write the PreToolUse JSON response to stdout."""
    sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", **extra}}))
    sys.stdout.flush()


def _handle_block(tool_name: str, wiki: str, rule: str, payload: EventPayload) -> None:
    """Emit a ``permissionDecision: deny`` response and tag the payload."""
    reason = (
        f"⚠ Library blocks this action: [[{wiki}]] - {rule} Confirm with the user before retrying."
    )
    _emit_hook_output({"permissionDecision": "deny", "permissionDecisionReason": reason})
    payload["blocked"] = True
    payload["severity"] = "block"
    payload["blocker_entry"] = wiki
    payload["summary"] = f"{tool_name} blocked by [[{wiki}]]"


def _handle_advise(tool_name: str, wiki: str, rule: str, payload: EventPayload) -> None:
    """Emit an ``additionalContext`` FYI response and tag the payload."""
    notice = f"📚 Library note (FYI; agent decides): [[{wiki}]] applies here — {rule}"
    _emit_hook_output({"additionalContext": notice})
    payload["blocked"] = False
    payload["severity"] = "advise"
    payload["blocker_entry"] = wiki
    payload["summary"] = f"{tool_name} noted by [[{wiki}]]"


def main() -> None:
    # Recursion guard — must come before any work. See module docstring
    # for why this is non-negotiable for daemon stability.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    # ── 1. Read stdin. Malformed input → no decision, no event. ────────
    try:
        hook_input = _read_hook_input()
    except (json.JSONDecodeError, ValueError, EOFError):
        return

    raw_tool_name: Any = hook_input.get("tool_name") or ""
    tool_name = raw_tool_name if isinstance(raw_tool_name, str) else str(raw_tool_name)
    raw_tool_input: Any = hook_input.get("tool_input") or {}
    tool_input: dict[str, Any] = (
        cast("dict[str, Any]", raw_tool_input) if isinstance(raw_tool_input, dict) else {}
    )

    # ── 2. Blocker check. Cached, sub-100ms even with hundreds of
    # entries (see _blockers.py for the cache strategy). ───────────────
    decision_payload: EventPayload = {"role": "assistant", "tool": tool_name, "phase": "pre"}
    knowledge_dir = get_config().knowledge_dir
    try:
        blockers = load_blockers(knowledge_dir)
        match = find_match(blockers, tool_name, tool_input)
    except Exception:
        # The blocker check must never crash the host agent. Worst case
        # is "no block" — same as today's behaviour without this tier.
        match = None

    if match is None:
        decision_payload["blocked"] = False
        decision_payload["summary"] = f"{tool_name} pre-check ok"
    else:
        rule = match.one_line_rule or "(no inline rule text)"
        try:
            wiki = rel_to_knowledge(match.entry_path, knowledge_dir)
        except Exception:
            wiki = match.entry_path.stem
        if match.severity == "block":
            _handle_block(tool_name, wiki, rule, decision_payload)
        else:
            _handle_advise(tool_name, wiki, rule, decision_payload)

    # ── 3. Event append (always, regardless of deny). The daemon needs
    # to see both the call and the block decision; PostToolUse won't
    # fire for denied calls so this is the only place to record them. ─
    try:
        append_event("PreToolUse", hook_input, payload=decision_payload)
    except Exception:
        # _events.append_event already swallows IO errors; this is a
        # belt-and-braces guard against an import-time failure.
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
