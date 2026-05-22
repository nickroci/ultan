"""SessionStart hook logic.

Injects knowledge base context into every conversation. When Claude
Code starts a session, this hook reads the knowledge base index and
recent daily log from the user-global store, then injects them as
additional context so Claude always "remembers" what it has learned.

This module holds the testable logic; ``session-start.py`` is the thin
shim Claude Code invokes via settings.json.
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

from _events import append_event  # noqa: E402
from aliases import session_bucket  # noqa: E402

MAX_CONTEXT_CHARS = 20_000
MAX_LOG_LINES = 30


def _store_dir() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}`` at call time."""
    override = os.environ.get("AGENT_MEM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".agent-mem"


def _ensure_store_dirs(store: Path) -> None:
    for sub in ("", "state", "knowledge", "daily"):
        try:
            (store / sub if sub else store).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def get_recent_log(daily_dir: Path) -> str:
    """Read the most recent daily log (today or yesterday)."""
    today = datetime.now(timezone.utc).astimezone()

    for offset in range(2):
        date = today - timedelta(days=offset)
        log_path = daily_dir / f"{date.strftime('%Y-%m-%d')}.md"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            recent = lines[-MAX_LOG_LINES:] if len(lines) > MAX_LOG_LINES else lines
            return "\n".join(recent)

    return "(no recent daily log)"


def build_context(project_slug: str, store: Path) -> str:
    """Assemble the context to inject into the conversation."""
    parts = []

    today = datetime.now(timezone.utc).astimezone()
    parts.append(
        f"## Today\n{today.strftime('%A, %B %d, %Y')}\n\n"
        f"## Current project\n`{project_slug}` "
        f"(lessons under `knowledge/projects/{project_slug}/` apply most directly)"
    )

    index_file = store / "knowledge" / "index.md"
    if index_file.exists():
        index_content = index_file.read_text(encoding="utf-8")
        parts.append(f"## Knowledge Base Index\n\n{index_content}")
    else:
        parts.append("## Knowledge Base Index\n\n(empty - no articles compiled yet)")

    recent_log = get_recent_log(store / "daily")
    parts.append(f"## Recent Daily Log\n\n{recent_log}")

    context = "\n\n---\n\n".join(parts)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...(truncated)"

    return context


def main() -> None:
    # Recursion guard. SessionStart fires on every Claude Code launch,
    # including the ones spawned by flush.py — without this, every
    # background flush would fire an extra SessionStart event.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    store = _store_dir()
    _ensure_store_dirs(store)

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
    # mid-conversation resume.
    source = hook_input.get("source") or "unknown"
    append_event("SessionStart", hook_input, payload={"source": str(source)})

    if not hook_cwd:
        hook_cwd = os.getcwd()

    from scope import current_project_slug  # local import keeps fast path lean

    project_slug = current_project_slug(hook_cwd)

    # Resolve which library bucket this session belongs to. Same call
    # the nudge filter (and eventually scholar's write path) uses — one
    # function answers "what bucket?" for every layer. Auto-creates the
    # alias entry inside session_bucket when there's evidence of a real
    # bucket. Wrapped in try/except because session-start MUST never
    # fail the host.
    try:
        session_bucket(store, Path(hook_cwd), project_slug)
    except Exception:
        pass

    context = build_context(project_slug, store)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":  # pragma: no cover
    main()
