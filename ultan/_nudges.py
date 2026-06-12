"""Pending-nudges consumer for the UserPromptSubmit hook (plugin path).

The Scholar (daemon side, ``agent_mem_daemon/scholar_prompt.py``) WRITES
ratified interrupts to ``${AGENT_MEM_HOME:-~/.agent-mem}/pending-nudges.md``
via ``append_nudges_from_response``. This module is the READ side: it is the
plugin/daemon-path port of the legacy ``src/hooks/_nudges.py`` consumer that
was dropped in the migration, so queued nudges stopped being delivered.

:func:`take_and_render` is the one entry point the hook calls each turn. It:

1. Reads and ATOMICALLY clears ``pending-nudges.md`` (renames it aside to a
   ``.consumed`` sibling — the cheapest atomic clear; keeps the last batch
   visible for debugging).
2. Parses the ``---``-delimited blocks (``id`` / ``created`` / ``lesson`` +
   body) into :class:`Nudge` objects.
3. Cross-project filter: nudges scoped to ``projects/<other>/...`` that don't
   match the current project slug are filtered out and RE-QUEUED to disk for a
   future session in that project. ``global/...`` and unrecognised buckets
   deliver to anyone. Slug resolution uses local stdlib mirrors of the
   ``aliases`` bucket helpers (see below) so this stays importable in the thin
   install; keep them byte-faithful to ``tools/search/aliases.py``.
4. Enforces a budget of 1 nudge/turn and 3/session, with the per-session
   counter persisted atomically at
   ``${AGENT_MEM_HOME}/state/nudge-budget-<session_id>.json``. The file is
   read+cleared even when the budget is exhausted, so nudges never accumulate
   unread forever.
5. Renders the survivors as ``additionalContext`` markdown.

Hot-path invariant: stdlib-only and fast (no LLM calls, just file I/O). The
project-bucket alias helpers are mirrored locally (see below) rather than
imported from ``aliases``, so this module stays importable in the thin
(no-``[retrieval]``) install. The file format is the single source of truth in
the Scholar's module docstring; the parser here mirrors it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

# Project-bucket alias helpers — shared local stdlib mirror (see _aliases.py),
# so this stays importable in the thin (no-[retrieval]) install.
from ._aliases import bucket_canonical_slug, load_aliases

# Pinned per-turn / per-session budgets (legacy parity, PLAN §5).
DEFAULT_PER_TURN_BUDGET = 1
DEFAULT_PER_SESSION_BUDGET = 3


# ── Path resolution ───────────────────────────────────────────────────


def _agent_mem_home() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}`` (mirrors ``_priming``)."""
    override = os.environ.get("AGENT_MEM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".agent-mem"


def pending_nudges_path() -> Path:
    return _agent_mem_home() / "pending-nudges.md"


def state_dir() -> Path:
    return _agent_mem_home() / "state"


def budget_state_path(session_id: str) -> Path:
    # Sanitise — session_ids are usually UUIDs but defend against weird
    # characters from synthetic inputs so we never escape the state dir.
    safe = "".join(c if (c.isalnum() or c in "._-") else "-" for c in session_id) or "unknown"
    return state_dir() / f"nudge-budget-{safe}.json"


# ── Nudge parsing ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Nudge:
    """One parsed nudge block."""

    id: str
    created: str
    lesson: str
    text: str


def parse_nudges(body: str) -> List[Nudge]:
    """Parse pending-nudges.md content into ordered :class:`Nudge` objects.

    Mirrors the Scholar's write format (single source of truth lives in
    ``agent_mem_daemon/scholar_prompt.py``). We can't import the daemon
    package here — the hook stays dependency-light — so the parser is
    duplicated, matching the legacy ``src/hooks/_nudges.py`` semantics.
    """
    if not body or not body.strip():
        return []
    out: List[Nudge] = []
    lines = body.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        while i < n and lines[i].strip() != "---":
            i += 1
        if i >= n:
            break
        i += 1  # past opening ---
        meta: Dict[str, str] = {}
        while i < n and lines[i].strip() != "---":
            line = lines[i]
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
            i += 1
        if i >= n:
            break
        i += 1  # past closing ---
        body_lines: List[str] = []
        while i < n and lines[i].strip() != "---":
            body_lines.append(lines[i])
            i += 1
        text = "\n".join(body_lines).strip()
        if meta or text:
            out.append(
                Nudge(
                    id=meta.get("id", ""),
                    created=meta.get("created", ""),
                    lesson=meta.get("lesson", ""),
                    text=text,
                )
            )
    return out


def _serialize_nudges(nudges: List[Nudge]) -> str:
    """Serialise back to pending-nudges.md wire format. Inverse of
    :func:`parse_nudges`."""
    chunks: List[str] = []
    for n in nudges:
        chunks.append(f"---\nid: {n.id}\ncreated: {n.created}\nlesson: {n.lesson}\n---\n{n.text}\n")
    return "\n".join(chunks)


# ── Budget bookkeeping ────────────────────────────────────────────────


def _load_consumed(session_id: str) -> int:
    """Return the per-session consumed count, or 0 on any read/decode
    failure (worst case: one extra nudge gets through)."""
    path = budget_state_path(session_id)
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
        return 0
    if not isinstance(data, dict):
        return 0
    data_map = cast("Mapping[str, Any]", data)
    try:
        return int(data_map.get("consumed", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _save_consumed(session_id: str, consumed: int, now: float) -> None:
    """Atomically persist the per-session counter via temp + rename.

    Persistence failure is non-fatal — worst case the same session shows
    one more nudge than its ceiling. We stay silent: hook stderr is
    surfaced to the user and noise would only confuse.
    """
    path = budget_state_path(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    data = {"consumed": int(consumed), "updated": now}
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
    except OSError:
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


# ── Nudge file consumption ────────────────────────────────────────────


def _read_and_clear_nudges_file(path: Path) -> str:
    """Read the nudges file and move it aside in one shot.

    Renames the file to ``<path>.consumed`` (only the most-recent consumed
    file is kept; older ones are overwritten). Renaming is the cheapest
    atomic clear — no truncate-then-write needed. Returns the body as a
    string, or empty string on any failure / empty file.
    """
    try:
        body = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    if not body:
        # File existed but is empty — still try to clean it up.
        try:
            path.unlink()
        except OSError:
            pass
        return ""
    consumed_path = path.with_suffix(path.suffix + ".consumed")
    try:
        # ``os.replace`` is atomic on POSIX; on Windows it overwrites any
        # existing target, which is what we want here too.
        os.replace(path, consumed_path)
    except OSError:
        # If we can't move it, try to truncate so we don't re-emit the same
        # nudges next turn.
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.truncate(0)
        except OSError:
            pass
    return body


def _requeue_nudges(path: Path, nudges: List[Nudge]) -> None:
    """Write ``nudges`` back to the pending-nudges file atomically.

    Called when cross-project nudges should be preserved for a future
    session in the matching project. Errors are swallowed — a broken
    re-queue must never crash the hook.
    """
    if not nudges:
        return
    body = _serialize_nudges(nudges)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(body)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    except OSError:
        pass


# ── Cross-project scope filter ────────────────────────────────────────


def _lesson_project_bucket(lesson_path: str) -> Optional[str]:
    """Extract the project bucket name from a lesson path.

    - ``projects/<bucket>/...`` -> ``"<bucket>"``
    - ``global/...``            -> ``"__global__"``
    - anything else             -> ``None`` (treated as universal so we
      never silently drop a nudge whose path layout we don't recognise).
    """
    if not lesson_path:
        return None
    parts = lesson_path.strip().lstrip("/").split("/")
    if len(parts) >= 2 and parts[0] == "projects":
        return parts[1] or None
    if parts and parts[0] == "global":
        return "__global__"
    return None


def _nudge_matches_project(
    nudge: Nudge,
    current_project_slug: Optional[str],
    aliases: Optional[Dict[str, str]] = None,
) -> bool:
    """True if ``nudge`` should be delivered to a session in the given
    project.

    Global and unrecognised-bucket nudges deliver to anyone; project-scoped
    nudges only deliver to the matching project. When the session has no
    project context (``current_project_slug`` is ``None``) we deliver
    everything — better to over-deliver than to lose a nudge that no future
    session can claim either.

    The bucket is resolved to its canonical slug (via the shared alias map)
    before comparison, so a folder named ``agent-mem`` can match a session
    slug like ``github.com-nickroci-ultan``. ``aliases`` is accepted so the
    caller can load it once and pass it in, avoiding a per-nudge file read.
    """
    bucket = _lesson_project_bucket(nudge.lesson)
    if bucket is None or bucket == "__global__":
        return True
    if not current_project_slug:
        return True
    if aliases is None:
        aliases = load_aliases(_agent_mem_home())
    return bucket_canonical_slug(bucket, aliases) == current_project_slug


# ── Consumption + budget enforcement ──────────────────────────────────


def take_nudges(
    session_id: str,
    *,
    per_turn_budget: int = DEFAULT_PER_TURN_BUDGET,
    per_session_budget: int = DEFAULT_PER_SESSION_BUDGET,
    current_project_slug: Optional[str] = None,
    now: Optional[float] = None,
) -> Tuple[List[Nudge], int]:
    """Pop up to ``per_turn_budget`` nudges for this turn.

    Honours the per-session ceiling: even if 100 nudges are queued, a
    session that has already shown ``per_session_budget`` returns ``([], N)``.

    Nudges whose ``lesson:`` path is scoped to a different project are
    filtered out and **re-queued** for a future session in the matching
    project. ``global/...`` and unrecognised-bucket nudges are delivered to
    any session.

    The file is ALWAYS read+cleared (even when the session budget is
    exhausted) so queued nudges never accumulate forever waiting for a
    session that has already declined them all — the budget is a per-session
    ceiling, not a queue.

    Returns ``(selected_nudges, consumed_after)`` — ``consumed_after`` is the
    session's running total post-emission.
    """
    if now is None:
        now = time.time()
    nudges_path = pending_nudges_path()

    body = _read_and_clear_nudges_file(nudges_path)
    all_queued = parse_nudges(body)

    # Cross-project filter: load the alias map once per call so we don't
    # re-read the file for every queued nudge.
    aliases = load_aliases(_agent_mem_home())
    queued: List[Nudge] = []
    requeue: List[Nudge] = []
    for nudge in all_queued:
        if _nudge_matches_project(nudge, current_project_slug, aliases):
            queued.append(nudge)
        else:
            requeue.append(nudge)
    if requeue:
        _requeue_nudges(nudges_path, requeue)

    consumed = _load_consumed(session_id)
    remaining_session = max(0, per_session_budget - consumed)
    take = min(per_turn_budget, remaining_session, len(queued))
    selected = queued[:take]
    # Anything beyond ``take`` is dropped — we already cleared the file so it
    # doesn't roll over. The Scholar vets aggressively, so overflow is rare.
    new_consumed = consumed + len(selected)

    if selected:
        _save_consumed(session_id, new_consumed, now)

    return selected, new_consumed


# ── Context rendering ─────────────────────────────────────────────────


def render_context(nudges: List[Nudge]) -> str:
    """Format selected nudges as the ``additionalContext`` body.

    Factual, present-tense, "the user has not been asked" — apply if
    relevant. One bullet per nudge: the lesson wikilink then the text.
    """
    if not nudges:
        return ""
    lines = [
        f"Memory has {len(nudges)} relevant lesson(s). "
        "Apply if relevant. The user has not been asked."
    ]
    for n in nudges:
        ref = f"[[{n.lesson}]]" if n.lesson else "(unknown lesson)"
        lines.append(f"- {ref}: {n.text}")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────


def take_and_render(session_id: str, project_slug: Optional[str]) -> str:
    """Consume pending nudges for this turn and render them to markdown.

    The single entry point the UserPromptSubmit hook calls. Reads + clears
    ``pending-nudges.md``, applies the cross-project filter (re-queuing
    foreign-project nudges), enforces the 1/turn + 3/session budget, and
    returns the rendered ``additionalContext`` body — or ``""`` when there
    is nothing to inject (no file, empty file, all filtered out, or budget
    exhausted).

    Never raises: a broken nudges file must never crash the host hook, so
    callers may invoke this without a try/except (it swallows its own I/O
    errors internally). ``session_id`` keys the per-session budget counter;
    ``project_slug`` is the current session's canonical project slug (the
    daemon path resolves it via ``aliases.session_bucket``) and gates the
    cross-project filter — pass ``None`` for a session with no project
    context to deliver everything.
    """
    if not session_id:
        # No session key → no budget bookkeeping possible. Skip entirely
        # rather than consume nudges we can't account against a session.
        return ""
    selected, _consumed = take_nudges(session_id, current_project_slug=project_slug)
    return render_context(selected)
