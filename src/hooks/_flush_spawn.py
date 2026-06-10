"""Shared helper for hooks that snapshot the conversation and hand off
to ``scripts/flush.py``.

``session-end.py`` and ``pre-compact.py`` had a near-identical body —
read the transcript, extract last-N turns of markdown, write a temp
context file under the store, spawn flush.py with the right args. The
only differences were:

- The minimum-turns threshold (1 for SessionEnd, 5 for PreCompact —
  pre-compact fires automatically and we don't want flush spam for tiny
  contexts).
- The temp-file naming prefix (so operators tailing the state dir can
  tell which hook produced the file).
- The log line tag.

Centralising the logic here means a future change to the transcript
schema or the flush argv only touches one place.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, cast

MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000


def _extract_role_and_content(entry: Mapping[str, Any]) -> tuple[str, object]:
    """Pull ``role`` and ``content`` from one transcript entry.

    Claude Code wraps the chat payload in ``message`` for most lines
    but writes some entries flat at the top level. Try the wrapped
    shape first, fall back to the flat one.
    """
    msg_any: Any = entry.get("message", {})
    if isinstance(msg_any, dict):
        msg = cast("Mapping[str, Any]", msg_any)
        role_any: Any = msg.get("role", "")
        content_any: Any = msg.get("content", "")
        return (role_any if isinstance(role_any, str) else ""), content_any
    role_flat: Any = entry.get("role", "")
    content_flat: Any = entry.get("content", "")
    return (role_flat if isinstance(role_flat, str) else ""), content_flat


def _content_to_text(content: object) -> str:
    """Flatten a content field to a plain string.

    Content can be a string or a list of blocks (text / tool_use /
    tool_result). We keep only text blocks; everything else is dropped.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    blocks = cast("list[Any]", content)
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            block_map = cast("Mapping[str, Any]", block)
            if block_map.get("type") == "text":
                text_any: Any = block_map.get("text", "")
                parts.append(text_any if isinstance(text_any, str) else "")
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def _iter_turns(transcript_path: Path) -> list[str]:
    """Read the JSONL transcript and return formatted turn strings."""
    turns: list[str] = []
    with open(transcript_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            role, content = _extract_role_and_content(entry)
            if role not in ("user", "assistant"):
                continue
            text = _content_to_text(content)
            if not text.strip():
                continue
            label = "User" if role == "user" else "Assistant"
            turns.append(f"**{label}:** {text.strip()}\n")
    return turns


def _trim_to_budget(context: str) -> str:
    """Cap ``context`` at ``MAX_CONTEXT_CHARS``, keeping a clean turn boundary."""
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    context = context[-MAX_CONTEXT_CHARS:]
    boundary = context.find("\n**")
    if boundary > 0:
        context = context[boundary + 1 :]
    return context


def extract_conversation_context(transcript_path: Path) -> tuple[str, int]:
    """Read JSONL transcript and extract last-N conversation turns as markdown.

    The transcript layout follows Claude Code's JSONL convention:
    one JSON object per line, with a top-level ``message`` dict whose
    ``role`` is ``user`` or ``assistant`` and whose ``content`` is
    either a string or a list of content blocks (text + tool_use +
    tool_result). We pull text-only blocks.

    Returns ``(markdown_string, recent_turn_count)``. An unreadable or
    empty transcript yields ``("", 0)``.
    """
    recent = _iter_turns(transcript_path)[-MAX_TURNS:]
    context = _trim_to_budget("\n".join(recent))
    return context, len(recent)


def snapshot_and_spawn_flush(
    transcript_path_str: str,
    session_id: str,
    project_slug: str,
    *,
    state_dir: Path,
    code_root: Path,
    min_turns: int,
    file_prefix: str,
    log_tag: str,
) -> Optional[Path]:
    """Read the transcript, snapshot to ``state_dir``, spawn flush.py.

    Returns the snapshot file path on success, ``None`` if anything
    short-circuited (no transcript, missing file, empty context, too
    few turns, or extraction error). Every short-circuit is logged but
    never raised — the host agent must never see a flush failure.
    """
    if not transcript_path_str:
        logging.info("SKIP[%s]: no transcript path", log_tag)
        return None

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        logging.info("SKIP[%s]: transcript missing: %s", log_tag, transcript_path_str)
        return None

    try:
        context, turn_count = extract_conversation_context(transcript_path)
    except Exception as e:
        logging.error("[%s] Context extraction failed: %s", log_tag, e)
        return None

    if not context.strip():
        logging.info("SKIP[%s]: empty context", log_tag)
        return None

    if turn_count < min_turns:
        logging.info("SKIP[%s]: only %d turns (min %d)", log_tag, turn_count, min_turns)
        return None

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    context_file = state_dir / f"{file_prefix}-{session_id}-{timestamp}.md"
    try:
        context_file.write_text(context, encoding="utf-8")
    except OSError as e:
        logging.error("[%s] Failed to write context file: %s", log_tag, e)
        return None

    flush_script = code_root / "scripts" / "flush.py"
    if not flush_script.exists():
        # Installed-wheel mode: config.CODE_ROOT resolves into site-packages'
        # parent, where there is no scripts/flush.py (or uv project) to spawn.
        # Flush is a repo-mode feature — degrade to a logged no-op instead of
        # pointing `uv run` at a lib directory.
        logging.info(
            "SKIP[%s]: flush.py not found at %s (installed-wheel mode)", log_tag, flush_script
        )
        return None
    cmd = [
        "uv",
        "run",
        "--directory",
        str(code_root),
        "python",
        str(flush_script),
        str(context_file),
        session_id,
        project_slug,
    ]

    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        logging.info(
            "[%s] Spawned flush.py for session %s project=%s (%d turns, %d chars)",
            log_tag,
            session_id,
            project_slug,
            turn_count,
            len(context),
        )
    except Exception as e:
        logging.error("[%s] Failed to spawn flush.py: %s", log_tag, e)
        return None

    return context_file
