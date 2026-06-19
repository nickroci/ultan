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
from pathlib import Path
from typing import Any, Optional, cast

from . import _blockers, _daemon, _events, _nudges, _priming, _session_context

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


def _project_slug_for_cwd(payload: dict[str, Any]) -> Optional[str]:
    """Canonical project slug for this turn — READ-ONLY on the hot path.

    Derives ``host/owner/repo`` (git remote → cwd basename → ``unknown``) via
    ``_session_context._project_slug``, the pure stdlib mirror of
    ``scope.current_project_slug``. CRITICAL: this must NOT trigger an alias-file
    write. ``aliases.session_bucket`` (which *can* bootstrap-write the alias map)
    is deliberately NOT called here — it runs once in ``_session_start`` only.
    Faithful to legacy ``src/hooks/user_prompt_submit.py``, which passed
    ``current_project_slug(cwd)`` (a pure slug) into the nudge filter and let
    ``session_start.py`` own the bootstrap.
    """
    cwd_raw: Any = payload.get("cwd")
    cwd_path = Path(cwd_raw).expanduser() if isinstance(cwd_raw, str) and cwd_raw else Path.cwd()
    # _project_slug never raises and never writes; returns "unknown" sentinel
    # rather than empty, matching scope.current_project_slug.
    return _session_context._project_slug(cwd_path)  # pyright: ignore[reportPrivateUsage]


def _user_prompt_submit(payload: dict[str, Any]) -> int:
    # Lazy-start the daemon if it's down; never blocks on its ~25s warmup —
    # we use the lexical fallback for this turn and the daemon is warm next time.
    _daemon.ensure_running()
    # Capture: record the prompt so the daemon's Librarian sees the turn's
    # intent. Independent of the stdout priming below (file write, not stdout).
    _events.append_event("UserPromptSubmit", payload, payload=_events.user_prompt_payload(payload))
    prompt = payload.get("prompt")
    session_id = payload.get("session_id")
    sid = session_id if isinstance(session_id, str) else None

    # Derive the canonical project slug ONCE, read-only, and reuse it for both
    # project-scoped priming (G6) and the cross-project nudge filter (G3).
    project_slug = _project_slug_for_cwd(payload)

    # G6: priming, now project-scoped so the daemon's scope-boost can prefer
    # this project's lessons.
    md = _priming.get_priming(
        prompt if isinstance(prompt, str) else "",
        project_slug=project_slug,
        session_id=sid,
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

    # Record WHAT priming surfaced this turn so the daemon's Librarian can spot
    # recall gaps — an entry relevant to the turn that did NOT surface — and
    # sharpen its retrieval triggers (retrieval-cue learning). File write only,
    # independent of the stdout priming below; a cheap regex over the rendered
    # markdown, so it stays within the per-turn hook budget.
    if sid:
        surfaced = _priming.extract_surfaced_links(md)
        if surfaced:
            _events.append_event(
                "Surfaced",
                payload,
                payload={
                    "role": "recall",
                    "content": "instant-recall surfaced: "
                    + ", ".join(f"[[{t}]]" for t in surfaced),
                },
            )

    # G3: consume + render any pending nudges for this turn (budget-gated,
    # cross-project filtered). Needs a session_id to key the per-session budget;
    # take_and_render no-ops to "" without one.
    nudges_md = _nudges.take_and_render(sid, project_slug) if sid else ""

    # Combine priming + nudges into a single additionalContext block, under an
    # `## Active nudges` header when both are present (the nudge agent's spec).
    parts: list[str] = []
    if md:
        parts.append(md)
    if nudges_md:
        if md:
            parts.append("---\n\n## Active nudges\n\n" + nudges_md)
        else:
            parts.append(nudges_md)
    # _emit_additional_context no-ops on empty, so a turn with neither emits
    # nothing.
    _emit_additional_context("UserPromptSubmit", "\n\n".join(parts))
    return 0


def _session_start(payload: dict[str, Any]) -> int:
    # If a daemon is still running OLDER code than what's now installed (e.g.
    # right after an `ultan` tool update), stop it so a fresh one starts on the
    # current code — a long-running process keeps the code it loaded at spawn.
    # Session-start, not the per-turn hot path, so the version check is fine here.
    _daemon.restart_if_stale()
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

    # cwd from the hook payload (NOT os.getcwd() — the hook process can run under
    # a different cwd than the active session). Used for the boot-context below.
    # NOTE: we deliberately do NOT bootstrap the project-aliases.json map here.
    # The daemon's Librarian already does it on every scan (librarian_prompt.
    # derive_project_bucket → aliases.session_bucket), so a SessionStart bootstrap
    # would be redundant — and pulling `aliases` into this module would break the
    # thin (no-`[retrieval]`) install, where the search package isn't present.
    cwd_raw: Any = payload.get("cwd")
    cwd = cwd_raw if isinstance(cwd_raw, str) else None

    # G5: inject the boot-context block (date + project + index head + recent
    # activity). The CURRENT bug is that _session_start returned 0 without ever
    # printing the additionalContext envelope, so SessionStart injection never
    # fired. _emit_additional_context prints the envelope and no-ops on empty.
    _emit_additional_context("SessionStart", _session_context.build_session_start_context(cwd))
    return 0


def _pre_tool_use(payload: dict[str, Any]) -> int:
    """PreToolUse: deterministic blocker check (Tier 3) + event capture.

    The only SYNCHRONOUS interrupt point Claude Code exposes — the hook fires
    before the tool runs, so a ``permissionDecision: deny`` refuses the call
    outright. Ported from legacy ``src/hooks/pre_tool_use.py``.

    stdin field names (confirmed against legacy pre_tool_use.py:111-116, which is
    the contract Claude Code sends for PreToolUse): ``tool_name`` (PascalCase,
    e.g. ``Bash``/``Edit``) and ``tool_input`` (the tool's argument dict, e.g.
    ``{"command": ...}`` for Bash, ``{"file_path": ...}`` for Edit).

    The recursion guard (``CLAUDE_INVOKED_BY``) lives inside ``_blockers.evaluate``
    so we don't re-implement it here; under that env var evaluate() returns None
    and we still record a benign event.
    """
    raw_tool_name: Any = payload.get("tool_name")
    tool_name = raw_tool_name if isinstance(raw_tool_name, str) else ""
    raw_tool_input: Any = payload.get("tool_input")
    tool_input: dict[str, Any] = (
        cast("dict[str, Any]", raw_tool_input) if isinstance(raw_tool_input, dict) else {}
    )

    decision = _blockers.evaluate(tool_name, tool_input)

    # Event payload tagging mirrors legacy pre_tool_use.py:120/131-132/83-96 so
    # the daemon's Librarian/Scholar see both the call and the block decision.
    event_payload: dict[str, Any] = {"role": "assistant", "tool": tool_name, "phase": "pre"}
    if decision is not None:
        print(
            json.dumps(
                {"hookSpecificOutput": {"hookEventName": "PreToolUse", **decision.hook_output}}
            )
        )
        event_payload["blocked"] = decision.severity == "block"
        event_payload["severity"] = decision.severity
        event_payload["blocker_entry"] = decision.wiki
        verb = "blocked" if decision.severity == "block" else "noted"
        event_payload["summary"] = f"{tool_name} {verb} by [[{decision.wiki}]]"
    else:
        event_payload["blocked"] = False
        event_payload["summary"] = f"{tool_name} pre-check ok"

    # ALWAYS append (even when denied): PostToolUse won't fire for a denied call,
    # so this is the only place a blocked call is recorded.
    _events.append_event("PreToolUse", payload, payload=event_payload)
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
    if event == "pre-tool-use":
        return _pre_tool_use(data)
    if event == "stop":
        # G1: forward transcript_path so the daemon's scheduler can read the
        # turn's assistant prose (scheduler._transcript_path_of pulls it off
        # ev.payload["transcript_path"]).
        _events.append_event("Stop", data, payload={"transcript_path": data.get("transcript_path")})
        return 0
    if event == "session-end":
        # G1: same — SessionEnd seals the final turn; carry transcript_path so
        # the closing turn's prose is captured too.
        _events.append_event(
            "SessionEnd", data, payload={"transcript_path": data.get("transcript_path")}
        )
        return 0
    if event == "pre-compact":
        # Legacy parity (src/hooks/pre_compact.py): emit a SessionEnd to force
        # a turn seal + Librarian pass BEFORE compaction discards transcript
        # detail. The type is SessionEnd on purpose — SessionEnd is the
        # daemon's turn-sealing path; it has no PreCompact handling.
        # G1: also carry transcript_path so the pre-compaction prose is captured
        # before compaction throws it away (legacy pre_compact flushed it).
        _events.append_event(
            "SessionEnd",
            data,
            payload={"source": "pre-compact", "transcript_path": data.get("transcript_path")},
        )
        return 0
    # Unreachable: every name in _EVENT_NAMES is handled above. Kept as a
    # fail-soft floor in case a new event name is added without a branch.
    return 0
