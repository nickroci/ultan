"""Read assistant prose out of the Claude Code transcript JSONL.

WHY THIS EXISTS
===============

The daemon's Librarian only ever sees ``events.jsonl`` — a stream of
``UserPromptSubmit`` + ``PostToolUse`` + ``Stop`` markers the hooks
append. That stream carries the user's prompts and the *names/arguments*
of tools the assistant ran, but it does NOT carry the assistant's
natural-language reasoning. The model's actual prose ("I'll use uv as
per your convention because…", "this contradicts the entry that says…")
is invisible to the Librarian, so a whole class of salient signal — the
assistant relying on, citing, or contradicting a memory in plain English
— never reaches the curator.

The LEGACY ``src/hooks/_flush_spawn.py`` path captured this: it read the
Claude Code transcript at ``transcript_path`` (present on the Stop /
SessionEnd hook stdin) and kept the ``type=="text"`` blocks from both
user and assistant messages. This module faithfully ports that
transcript-parsing logic for the daemon, narrowed to the assistant's
prose (the user's prompts already arrive via ``UserPromptSubmit``), and
adds an offset/marker so repeated reads of the same transcript don't
re-surface turns the Librarian already saw.

DESIGN
======

* **Stdlib only.** ``json`` + ``pathlib`` — same constraint the rest of
  the ingest path honours.
* **Faithful port.** :func:`_content_to_text` and the role/message
  unwrapping mirror ``_flush_spawn._content_to_text`` /
  ``_extract_role_and_content`` exactly (markdown-aware: keep ``text``
  blocks, drop ``tool_use`` / ``tool_result``).
* **Incremental.** Each transcript line in the Claude Code JSONL carries
  a stable ``uuid``. We persist a per-session high-water set of seen
  uuids (and a line offset as a fast-path) so a Stop that fires twice on
  a growing transcript only yields the NEW assistant turns. Keyed per
  session in a small JSON state file, following the
  ``ingest._load_offset_state`` / ``_save_offset_state`` idiom
  (atomic tmp+rename).
* **Fail-soft.** A missing, unreadable, or malformed transcript yields
  an empty list and logs at DEBUG — it must NEVER crash the daemon. A
  Stop with no prose is the common, expected case.
* **Bounded.** Capped at :data:`MAX_PROSE_TURNS` newest turns per read,
  mirroring the legacy ``MAX_TURNS`` guard, so a huge transcript can't
  blow up the Librarian prompt.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

log = logging.getLogger("agent_mem_daemon.transcript")

# Cap on how many assistant-prose turns we hand back from a single read.
# Mirrors ``_flush_spawn.MAX_TURNS`` — a runaway transcript must not push
# the Librarian prompt past Haiku's context budget. We keep the NEWEST
# turns (the recent reasoning is what matters for salience).
MAX_PROSE_TURNS = 30

# Per-prose-turn character cap. A single assistant message can be huge
# (the model occasionally dumps a multi-KB plan). Trim each so one turn
# can't dominate the buffer; the librarian prompt has its own recency
# cap downstream, but bounding here keeps the synthetic events sane.
MAX_PROSE_CHARS = 4_000

# How many seen-uuids we retain per session in the marker file. A session
# rarely has more than a few hundred assistant turns; this bounds the
# state file so it can't grow without limit on a marathon session.
MAX_SEEN_UUIDS_PER_SESSION = 2_000


@dataclass
class ProseTurn:
    """One assistant natural-language turn pulled from the transcript.

    ``uuid`` is the transcript line's stable id (used for dedup across
    repeated reads); ``text`` is the flattened markdown prose.
    """

    uuid: str
    text: str


# ── Transcript parsing (ported from src/hooks/_flush_spawn.py) ──────────


def _extract_role_and_content(entry: Mapping[str, Any]) -> tuple[str, object]:
    """Pull ``role`` and ``content`` from one transcript entry.

    Claude Code wraps the chat payload in ``message`` for most lines but
    writes some entries flat at the top level. Try the wrapped shape
    first, fall back to the flat one. Faithful port of
    ``_flush_spawn._extract_role_and_content``.
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
    """Flatten a content field to a plain string, keeping only text blocks.

    Content can be a string or a list of blocks (text / tool_use /
    tool_result). We keep only ``type == "text"`` blocks; tool calls and
    tool results are dropped (the daemon already sees those as
    ``PostToolUse`` events). Faithful port of
    ``_flush_spawn._content_to_text``.
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


def _entry_uuid(entry: Mapping[str, Any], *, line_no: int) -> str:
    """Best-effort stable id for a transcript line.

    Claude Code stamps every transcript line with a top-level ``uuid``.
    We fall back to the 1-based line number when a line lacks one so a
    malformed/legacy transcript still dedups (within a single file) — the
    line number is stable for an append-only transcript.
    """
    raw = entry.get("uuid")
    if isinstance(raw, str) and raw:
        return raw
    return f"line:{line_no}"


def iter_assistant_prose(transcript_path: Path) -> List[ProseTurn]:
    """Parse ``transcript_path`` and return every assistant text turn.

    Returns turns in file order (oldest first). Only ``assistant``-role
    messages with non-empty text survive — user prompts already reach the
    daemon via ``UserPromptSubmit`` events, so re-ingesting them here
    would double-count. An unreadable/missing transcript returns ``[]``
    and logs at DEBUG (fail-soft — never raises).
    """
    turns: List[ProseTurn] = []
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    loaded: object = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(loaded, dict):
                    continue
                entry = cast("Mapping[str, Any]", loaded)
                role, content = _extract_role_and_content(entry)
                if role != "assistant":
                    continue
                text = _content_to_text(content).strip()
                if not text:
                    continue
                if len(text) > MAX_PROSE_CHARS:
                    text = text[:MAX_PROSE_CHARS] + "\n... (truncated)"
                turns.append(ProseTurn(uuid=_entry_uuid(entry, line_no=line_no), text=text))
    except (FileNotFoundError, IsADirectoryError):
        log.debug("transcript not found / not a file: %s", transcript_path)
        return []
    except OSError as e:
        log.debug("could not read transcript %s: %s", transcript_path, e)
        return []
    return turns


# ── Per-session marker state (dedup across repeated reads) ──────────────


def _load_marker_state(marker_path: Path) -> Dict[str, List[str]]:
    """Read the per-session seen-uuid map. Returns ``{}`` on any problem.

    Shape: ``{session_id: [seen_uuid, ...]}``. Mismatched/legacy shapes
    are tolerated by returning an empty map (we simply re-read the
    transcript, which is safe — the buffer already dedups by turn).
    """
    try:
        loaded: object = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    raw = cast(Dict[str, Any], loaded)
    out: Dict[str, List[str]] = {}
    for sid, seen in raw.items():
        if isinstance(seen, list):
            out[str(sid)] = [str(u) for u in cast("list[Any]", seen)]
    return out


def _save_marker_state(marker_path: Path, state: Mapping[str, List[str]]) -> None:
    """Persist the seen-uuid map atomically (tmp+rename). Best-effort —
    a write failure degrades to potential re-reads, never a crash."""
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, marker_path)
    except OSError as e:
        log.debug("could not persist transcript marker to %s: %s", marker_path, e)


def read_new_assistant_prose(
    transcript_path: Path,
    *,
    session_id: str,
    marker_path: Optional[Path] = None,
    max_turns: int = MAX_PROSE_TURNS,
) -> List[ProseTurn]:
    """Return the assistant-prose turns NOT yet seen for ``session_id``.

    This is the daemon-facing entry point. It:

    1. Parses the whole transcript (fail-soft → ``[]``).
    2. If ``marker_path`` is given, drops turns whose uuid was already
       recorded for this session, then records the newly-returned uuids
       so the next call won't re-surface them.
    3. Caps the result to the ``max_turns`` NEWEST unseen turns.

    Passing ``marker_path=None`` disables dedup (every call returns the
    full transcript) — used by callers that manage their own dedup or by
    tests. With a marker path, a Stop that fires twice on the same
    growing transcript yields only the genuinely new reasoning.
    """
    all_turns = iter_assistant_prose(transcript_path)
    if not all_turns:
        return []

    if marker_path is None:
        # No persistence requested — still honour the recency cap so a
        # huge transcript can't flood the buffer in one shot.
        return all_turns[-max_turns:]

    state = _load_marker_state(marker_path)
    seen = set(state.get(session_id, []))
    fresh = [t for t in all_turns if t.uuid not in seen]
    if not fresh:
        return []

    # Keep only the newest unseen turns for the buffer, but mark ALL fresh
    # uuids as seen so we don't re-surface the dropped-old ones next time.
    selected = fresh[-max_turns:]

    updated_seen = list(state.get(session_id, []))
    updated_seen.extend(t.uuid for t in fresh)
    if len(updated_seen) > MAX_SEEN_UUIDS_PER_SESSION:
        updated_seen = updated_seen[-MAX_SEEN_UUIDS_PER_SESSION:]
    state[session_id] = updated_seen
    _save_marker_state(marker_path, state)

    return selected
