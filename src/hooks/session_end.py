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

import json
import logging
import os
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _events import append_event  # noqa: E402
from _flush_spawn import snapshot_and_spawn_flush  # noqa: E402

MIN_TURNS_TO_FLUSH = 1


def _store_dir() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}`` at call time."""
    override = os.environ.get("AGENT_MEM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".agent-mem"


def _ensure_store_dirs(store: Path) -> None:
    """Best-effort mkdir of the store + state dir. Failures are
    swallowed; downstream writes already tolerate missing parents."""
    for sub in ("", "state", "knowledge", "daily"):
        try:
            (store / sub if sub else store).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def _setup_logging(store: Path) -> None:
    """Idempotent logging setup. Tests can monkeypatch
    ``logging.basicConfig`` to a no-op; production hooks call this once
    per process."""
    try:
        logging.basicConfig(
            filename=str(store / "flush.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [hook] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    except OSError:
        # Disk full / permission denied / parent missing. Logging is
        # nice-to-have; a missing flush.log must not break the hook.
        pass


def main() -> None:
    # Recursion guard: if we were spawned by flush.py (which calls
    # Agent SDK, which runs Claude Code, which would fire this hook
    # again), exit immediately.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    store = _store_dir()
    _ensure_store_dirs(store)
    _setup_logging(store)

    # Read hook input from stdin.
    # Claude Code on Windows may pass paths with unescaped backslashes;
    # the regex below escapes lone ones before retrying the JSON parse.
    try:
        raw_input = sys.stdin.read()
        try:
            hook_input: dict = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw_input)
            hook_input = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError) as e:
        logging.error("Failed to parse stdin: %s", e)
        return

    if not isinstance(hook_input, dict):
        return

    session_id = hook_input.get("session_id", "unknown")
    # NB: Claude Code's SessionEnd payload uses ``reason``, not
    # ``source``. Accept both for backwards-compat and synthetic test
    # events.
    source = hook_input.get("source") or hook_input.get("reason") or "unknown"
    transcript_path_str = hook_input.get("transcript_path", "")
    hook_cwd = hook_input.get("cwd") or os.getcwd()

    # Daemon event: emit BEFORE the flush.py spawn so a slow disk
    # doesn't delay the daemon's view of session end. Payload is empty
    # per PLAN §7 item 4 — turn boundary only.
    append_event("SessionEnd", hook_input, payload={})

    # Project slug is derived from the host agent's cwd, NOT this
    # hook's own cwd. Pass it explicitly to current_project_slug — the
    # hook may be invoked from anywhere depending on shell wrapping.
    from scope import current_project_slug  # local import: cheap, keeps top fast

    project_slug = current_project_slug(hook_cwd)

    logging.info(
        "SessionEnd fired: session=%s source=%s project=%s",
        session_id,
        source,
        project_slug,
    )

    snapshot_and_spawn_flush(
        transcript_path_str if isinstance(transcript_path_str, str) else "",
        str(session_id),
        project_slug,
        state_dir=store / "state",
        code_root=_CODE_ROOT,
        min_turns=MIN_TURNS_TO_FLUSH,
        file_prefix="session-flush",
        log_tag="session-end",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
