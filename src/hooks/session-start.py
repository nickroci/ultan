"""
SessionStart hook — injects knowledge base context into every conversation.

This is the "context injection" layer. When Claude Code starts a session,
this hook reads the knowledge base index and recent daily log from the
user-global store, then injects them as additional context so Claude always
"remembers" what it has learned.

Configure in .claude/settings.json (see _dot_claude_disabled/settings.json):
{
    "hooks": {
        "SessionStart": [{
            "matcher": "",
            "command": "uv run --directory /path/to/agent-mem/src python hooks/session-start.py"
        }]
    }
}
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import (  # noqa: E402
    DAILY_DIR,
    INDEX_FILE,
    ensure_store_dirs,
)
from scope import current_project_slug  # noqa: E402

from _events import append_event  # noqa: E402

ensure_store_dirs()

MAX_CONTEXT_CHARS = 20_000
MAX_LOG_LINES = 30


def get_recent_log() -> str:
    """Read the most recent daily log (today or yesterday)."""
    today = datetime.now(timezone.utc).astimezone()

    for offset in range(2):
        date = today - timedelta(days=offset)
        log_path = DAILY_DIR / f"{date.strftime('%Y-%m-%d')}.md"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            # Return last N lines to keep context small
            recent = lines[-MAX_LOG_LINES:] if len(lines) > MAX_LOG_LINES else lines
            return "\n".join(recent)

    return "(no recent daily log)"


def build_context(project_slug: str) -> str:
    """Assemble the context to inject into the conversation."""
    parts = []

    # Today's date + current project scope
    today = datetime.now(timezone.utc).astimezone()
    parts.append(
        f"## Today\n{today.strftime('%A, %B %d, %Y')}\n\n"
        f"## Current project\n`{project_slug}` "
        f"(lessons under `knowledge/projects/{project_slug}/` apply most directly)"
    )

    # Knowledge base index (the core retrieval mechanism)
    if INDEX_FILE.exists():
        index_content = INDEX_FILE.read_text(encoding="utf-8")
        parts.append(f"## Knowledge Base Index\n\n{index_content}")
    else:
        parts.append("## Knowledge Base Index\n\n(empty - no articles compiled yet)")

    # Recent daily log
    recent_log = get_recent_log()
    parts.append(f"## Recent Daily Log\n\n{recent_log}")

    context = "\n\n---\n\n".join(parts)

    # Truncate if too long
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...(truncated)"

    return context


def main():
    # Recursion guard. SessionStart fires on every Claude Code launch,
    # including the ones spawned by flush.py — without this, every
    # background flush would fire an extra SessionStart event into the
    # daemon's stream. Mirrors the guard in session-end.py / pre-compact.py.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    # Read hook input if present; fall back to environment otherwise.
    hook_input: dict = {}
    hook_cwd = None
    try:
        raw = sys.stdin.read()
        if raw.strip():
            hook_input = json.loads(raw)
            if not isinstance(hook_input, dict):
                hook_input = {}
            hook_cwd = hook_input.get("cwd")
    except (json.JSONDecodeError, ValueError, OSError):
        hook_input = {}

    # Daemon event. ``source`` comes from Claude Code's SessionStart
    # payload — one of startup|resume|clear|compact — and is the only
    # bit of state the daemon needs to distinguish a fresh boot from a
    # mid-conversation resume. ``append_event`` no-ops gracefully if
    # ``session_id`` is missing (e.g. user invoked the hook by hand for
    # testing) so we don't need to gate the call.
    source = hook_input.get("source") or "unknown"
    append_event("SessionStart", hook_input, payload={"source": str(source)})

    if not hook_cwd:
        hook_cwd = os.getcwd()

    project_slug = current_project_slug(hook_cwd)

    context = build_context(project_slug)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
