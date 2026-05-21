"""PreToolUse hook — synchronous deterministic blocker check + event stream.

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
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Recursion guard FIRST — before any imports or work. See module
# docstring for why this is non-negotiable for daemon stability.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _blockers import find_match, load_blockers, rel_to_knowledge  # noqa: E402
from _events import append_event  # noqa: E402


def _knowledge_dir() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}/knowledge``.

    Reuses ``config.KNOWLEDGE_DIR`` so the env-var override logic stays
    in one place. Importing inline keeps a config import failure from
    breaking the recursion-guard short-circuit at module load.
    """
    from config import KNOWLEDGE_DIR  # noqa: E402

    return KNOWLEDGE_DIR


def _read_hook_input() -> dict:
    """Parse the stdin JSON Claude Code sends, with the same Windows
    backslash workaround used by the other hooks.
    """
    raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Lone backslashes in file paths sometimes break strict JSON
        # parsing on Windows; double them up and retry.
        fixed = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw)
        return json.loads(fixed)


def main() -> None:
    # ── 1. Read stdin. Malformed input → no decision, no event. ────────
    try:
        hook_input = _read_hook_input()
    except (json.JSONDecodeError, ValueError, EOFError):
        return
    if not isinstance(hook_input, dict):
        return

    tool_name = hook_input.get("tool_name") or ""
    tool_input = hook_input.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    # ── 2. Blocker check. Cached, sub-100ms even with hundreds of
    # entries (see _blockers.py for the cache strategy). ───────────────
    decision_payload: dict = {"role": "assistant", "tool": str(tool_name), "phase": "pre"}
    denied = False
    try:
        knowledge_dir = _knowledge_dir()
        blockers = load_blockers(knowledge_dir)
        match = find_match(blockers, str(tool_name), tool_input)
    except Exception:
        # The blocker check must never crash the host agent. Worst case
        # is "no block" — same as today's behaviour without this tier.
        match = None
        blockers = []

    if match is not None:
        rule = match.one_line_rule or "(no inline rule text)"
        try:
            wiki = rel_to_knowledge(match.entry_path, knowledge_dir)
        except Exception:
            wiki = match.entry_path.stem

        if match.severity == "block":
            # Opt-in hard stop. Reserved for genuinely dangerous actions
            # the user has explicitly chosen to block (rm -rf, force-push
            # to main, drop prod db, etc.).
            denied = True
            reason = (
                f"⚠ Library blocks this action: "
                f"[[{wiki}]] - {rule} "
                f"Confirm with the user before retrying."
            )
            sys.stdout.write(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": reason,
                        }
                    }
                )
            )
            sys.stdout.flush()
            decision_payload["blocked"] = True
            decision_payload["severity"] = "block"
            decision_payload["blocker_entry"] = wiki
            decision_payload["summary"] = f"{tool_name} blocked by [[{wiki}]]"
        else:
            # Default: advisory FYI. Tool proceeds; the agent gets a
            # system reminder via additionalContext and decides what
            # to do. Like a human noticing a relevant memory mid-action
            # — not paralysing, just informing.
            notice = f"📚 Library note (FYI; agent decides): [[{wiki}]] applies here — {rule}"
            sys.stdout.write(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "additionalContext": notice,
                        }
                    }
                )
            )
            sys.stdout.flush()
            decision_payload["blocked"] = False
            decision_payload["severity"] = "advise"
            decision_payload["blocker_entry"] = wiki
            decision_payload["summary"] = f"{tool_name} noted by [[{wiki}]]"
    else:
        decision_payload["blocked"] = False
        decision_payload["summary"] = f"{tool_name} pre-check ok"

    # ── 3. Event append (always, regardless of deny). The daemon needs
    # to see both the call and the block decision; PostToolUse won't
    # fire for denied calls so this is the only place to record them. ─
    try:
        append_event("PreToolUse", hook_input, payload=decision_payload)
    except Exception:
        # _events.append_event already swallows IO errors; this is a
        # belt-and-braces guard against an import-time failure.
        pass

    if denied:
        # Claude Code expects exit code 0 for a structured JSON response
        # (non-zero is reserved for "hook itself crashed"). The deny
        # decision lives in the JSON body we already wrote.
        return


if __name__ == "__main__":
    main()
