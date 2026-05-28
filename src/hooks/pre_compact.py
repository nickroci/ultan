"""PreCompact hook logic.

When Claude Code's context window fills up, it auto-compacts (summarises
and discards detail). This hook fires BEFORE that happens, extracting
conversation context and spawning flush.py to extract knowledge that
would otherwise be lost to summarisation.

The hook itself does NO API calls — only local file I/O for speed
(<10s).

This module holds the testable logic; ``pre-compact.py`` is the thin
shim Claude Code invokes via settings.json.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _events import append_event  # noqa: E402
from _flush_spawn import snapshot_and_spawn_flush  # noqa: E402
from _hookutil import ensure_store_dirs, parse_stdin, setup_logging  # noqa: E402
from config import get_config  # noqa: E402
from scope import current_project_slug  # noqa: E402

MIN_TURNS_TO_FLUSH = 5


def main() -> None:
    # Recursion guard
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    store = get_config().store_dir
    ensure_store_dirs(store)
    setup_logging(store, "pre-compact")

    hook_input = parse_stdin()
    if hook_input is None:
        logging.error("Failed to parse stdin")
        return

    raw_session: Any = hook_input.get("session_id", "unknown")
    session_id: str = raw_session if isinstance(raw_session, str) else str(raw_session)
    raw_transcript: Any = hook_input.get("transcript_path", "")
    transcript_path_str: str = raw_transcript if isinstance(raw_transcript, str) else ""
    raw_cwd: Any = hook_input.get("cwd")
    hook_cwd: str = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else os.getcwd()

    project_slug = current_project_slug(hook_cwd)

    # Daemon event: PreCompact is treated as a SessionEnd-like signal so
    # the daemon's existing turn-sealing path applies. Per the task spec,
    # we emit ``type: "SessionEnd"`` with ``payload: {"source":
    # "pre-compact"}`` rather than inventing a new type the daemon
    # doesn't know about.
    append_event("SessionEnd", hook_input, payload={"source": "pre-compact"})

    logging.info("PreCompact fired: session=%s project=%s", session_id, project_slug)

    snapshot_and_spawn_flush(
        transcript_path_str,
        session_id,
        project_slug,
        state_dir=store / "state",
        code_root=_CODE_ROOT,
        min_turns=MIN_TURNS_TO_FLUSH,
        file_prefix="flush-context",
        log_tag="pre-compact",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
