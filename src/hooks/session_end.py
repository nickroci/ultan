"""SessionEnd hook logic.

When a Claude Code session ends, this hook reads the transcript path
from stdin, extracts conversation context, and spawns flush.py as a
background process to extract knowledge into the daily log.

The hook itself does NO API calls — only local file I/O for speed
(<10s).

Storage layout (under ``~/.agent-mem/``) is owned by config.py; this
hook only writes the temporary context file into that store and walks
away.

This module holds the testable logic; ``session-end.py`` is the thin
shim Claude Code invokes via settings.json.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from _events import append_event
from _flush_spawn import snapshot_and_spawn_flush
from _hookutil import ensure_store_dirs, parse_stdin, setup_logging
from config import CODE_ROOT, get_config
from scope import current_project_slug

MIN_TURNS_TO_FLUSH = 1


def main() -> None:
    # Recursion guard: if we were spawned by flush.py (which calls
    # Agent SDK, which runs Claude Code, which would fire this hook
    # again), exit immediately.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    store = get_config().store_dir
    ensure_store_dirs(store)
    setup_logging(store, "session-end")

    hook_input = parse_stdin()
    if hook_input is None:
        logging.error("Failed to parse stdin")
        return

    raw_session: Any = hook_input.get("session_id", "unknown")
    session_id: str = raw_session if isinstance(raw_session, str) else str(raw_session)
    # NB: Claude Code's SessionEnd payload uses ``reason``, not
    # ``source``. Accept both for backwards-compat and synthetic test
    # events.
    raw_source: Any = hook_input.get("source") or hook_input.get("reason") or "unknown"
    source: str = raw_source if isinstance(raw_source, str) else str(raw_source)
    raw_transcript: Any = hook_input.get("transcript_path", "")
    transcript_path_str: str = raw_transcript if isinstance(raw_transcript, str) else ""
    raw_cwd: Any = hook_input.get("cwd")
    hook_cwd: str = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else os.getcwd()

    # Daemon event: emit BEFORE the flush.py spawn so a slow disk
    # doesn't delay the daemon's view of session end. Payload is empty
    # per PLAN §7 item 4 — turn boundary only.
    append_event("SessionEnd", hook_input, payload={})

    # Project slug is derived from the host agent's cwd, NOT this
    # hook's own cwd. Pass it explicitly to ``current_project_slug``
    # — the hook may be invoked from anywhere depending on shell
    # wrapping.
    project_slug = current_project_slug(hook_cwd)

    logging.info(
        "SessionEnd fired: session=%s source=%s project=%s",
        session_id,
        source,
        project_slug,
    )

    snapshot_and_spawn_flush(
        transcript_path_str,
        session_id,
        project_slug,
        state_dir=store / "state",
        code_root=CODE_ROOT,
        min_turns=MIN_TURNS_TO_FLUSH,
        file_prefix="session-flush",
        log_tag="session-end",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
