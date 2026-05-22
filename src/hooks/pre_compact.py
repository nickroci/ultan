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

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, cast

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import scope  # noqa: E402
from _events import HookPayload, append_event  # noqa: E402
from _flush_spawn import snapshot_and_spawn_flush  # noqa: E402

# Local re-typed view of ``scope.current_project_slug`` — see
# ``user_prompt_submit.py`` for the rationale. The upstream signature
# still uses untyped ``os.PathLike`` so its inferred type leaks
# Unknown; ``getattr`` + ``cast`` recovers a clean ``str | None ->
# str`` callable without touching scope.py (owned by another slice).
_current_project_slug: Callable[[Optional[str]], str] = cast(
    Callable[[Optional[str]], str],
    getattr(scope, "current_project_slug"),
)

MIN_TURNS_TO_FLUSH = 5


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


def _setup_logging(store: Path) -> None:
    try:
        logging.basicConfig(
            filename=str(store / "flush.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [pre-compact] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    except OSError:
        pass


def main() -> None:
    # Recursion guard
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    store = _store_dir()
    _ensure_store_dirs(store)
    _setup_logging(store)

    try:
        raw_input = sys.stdin.read()
        parsed: Any
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw_input)
            parsed = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError) as e:
        logging.error("Failed to parse stdin: %s", e)
        return

    if not isinstance(parsed, dict):
        return
    hook_input: HookPayload = cast("HookPayload", parsed)

    raw_session: Any = hook_input.get("session_id", "unknown")
    session_id: str = raw_session if isinstance(raw_session, str) else str(raw_session)
    raw_transcript: Any = hook_input.get("transcript_path", "")
    transcript_path_str: str = raw_transcript if isinstance(raw_transcript, str) else ""
    raw_cwd: Any = hook_input.get("cwd")
    hook_cwd: str = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else os.getcwd()

    project_slug = _current_project_slug(hook_cwd)

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
