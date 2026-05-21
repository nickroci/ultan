"""
SessionEnd hook — captures conversation transcript for memory extraction.

When a Claude Code session ends, this hook reads the transcript path from
stdin, extracts conversation context, and spawns flush.py as a background
process to extract knowledge into the daily log.

The hook itself does NO API calls — only local file I/O for speed (<10s).

Storage layout (under ``~/.agent-mem/``) is owned by config.py; this hook
only writes the temporary context file into that store and walks away.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Recursion guard: if we were spawned by flush.py (which calls Agent SDK,
# which runs Claude Code, which would fire this hook again), exit immediately.
# This MUST happen before any heavy imports — see flush.py for the matching
# set of CLAUDE_INVOKED_BY at the top of that file.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

# Make scripts/ importable so we can pick up config + scope helpers.
_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _events import append_event  # noqa: E402
from config import (  # noqa: E402
    CODE_ROOT,
    STATE_DIR,
    STORE_DIR,
    ensure_store_dirs,
)
from scope import current_project_slug  # noqa: E402

ensure_store_dirs()

logging.basicConfig(
    filename=str(STORE_DIR / "flush.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [hook] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
MIN_TURNS_TO_FLUSH = 1


def extract_conversation_context(transcript_path: Path) -> tuple[str, int]:
    """Read JSONL transcript and extract last ~N conversation turns as markdown."""
    turns: list[str] = []

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = entry.get("role", "")
                content = entry.get("content", "")

            if role not in ("user", "assistant"):
                continue

            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)

            if isinstance(content, str) and content.strip():
                label = "User" if role == "user" else "Assistant"
                turns.append(f"**{label}:** {content.strip()}\n")

    recent = turns[-MAX_TURNS:]
    context = "\n".join(recent)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
        boundary = context.find("\n**")
        if boundary > 0:
            context = context[boundary + 1 :]

    return context, len(recent)


def main() -> None:
    # Read hook input from stdin.
    # Claude Code on Windows may pass paths with unescaped backslashes; the
    # regex below escapes lone ones before retrying the JSON parse.
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

    session_id = hook_input.get("session_id", "unknown")
    # NB: Claude Code's SessionEnd payload uses ``reason``, not ``source``.
    # We accept both for backwards-compat with older versions and for the
    # synthetic events the hook author may inject during testing.
    source = hook_input.get("source") or hook_input.get("reason") or "unknown"
    transcript_path_str = hook_input.get("transcript_path", "")
    hook_cwd = hook_input.get("cwd") or os.getcwd()

    # Daemon event: emit BEFORE the flush.py spawn so a slow disk doesn't
    # delay the daemon's view of session end. Payload is empty per PLAN
    # §7 item 4 — turn boundary only; full content stays in the transcript.
    append_event("SessionEnd", hook_input, payload={})

    # Project slug is derived from the host agent's cwd, NOT this hook's own
    # cwd. Pass it explicitly to current_project_slug — the hook may be
    # invoked from anywhere depending on shell wrapping.
    project_slug = current_project_slug(hook_cwd)

    logging.info(
        "SessionEnd fired: session=%s source=%s project=%s",
        session_id,
        source,
        project_slug,
    )

    if not transcript_path_str or not isinstance(transcript_path_str, str):
        logging.info("SKIP: no transcript path")
        return

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        logging.info("SKIP: transcript missing: %s", transcript_path_str)
        return

    # Extract conversation context in the hook (fast, no API calls)
    try:
        context, turn_count = extract_conversation_context(transcript_path)
    except Exception as e:
        logging.error("Context extraction failed: %s", e)
        return

    if not context.strip():
        logging.info("SKIP: empty context")
        return

    if turn_count < MIN_TURNS_TO_FLUSH:
        logging.info("SKIP: only %d turns (min %d)", turn_count, MIN_TURNS_TO_FLUSH)
        return

    # Write context to a temp file for the background process. State dir
    # lives in the user-global store, not in the code tree, so multiple
    # checkouts of agent-mem all share the same scratch space.
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    context_file = STATE_DIR / f"session-flush-{session_id}-{timestamp}.md"
    context_file.write_text(context, encoding="utf-8")

    # Spawn flush.py as a background process. Note: cwd / --directory is the
    # CODE root (where pyproject.toml lives), NOT the store.
    flush_script = _SCRIPTS_DIR / "flush.py"

    cmd = [
        "uv",
        "run",
        "--directory",
        str(CODE_ROOT),
        "python",
        str(flush_script),
        str(context_file),
        session_id,
        project_slug,
    ]

    # On Windows, use CREATE_NO_WINDOW to avoid flash console window.
    # Do NOT use DETACHED_PROCESS — it breaks the Agent SDK's subprocess I/O.
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        logging.info(
            "Spawned flush.py for session %s project=%s (%d turns, %d chars)",
            session_id,
            project_slug,
            turn_count,
            len(context),
        )
    except Exception as e:
        logging.error("Failed to spawn flush.py: %s", e)


if __name__ == "__main__":
    main()
