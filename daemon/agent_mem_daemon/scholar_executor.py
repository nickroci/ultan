"""Deterministic executor for the Scholar's validated ``ScholarDecisions``.

In the Pydantic-AI architecture the Scholar no longer touches files; it
RETURNS a typed, boundary-validated ``ScholarDecisions`` and this module
applies it to disk. Each action type maps to a small, failure-tolerant
handler that:

  - writes / updates / merges / moves / archives / deprecates the entry,
  - keeps ``knowledge/index.md`` in sync (one catalog row per live entry),
  - appends an audit block to ``knowledge/log.md``.

Moves reuse ``library_tools`` (the same atomic move-and-rewrite-wikilinks
machinery the old Scholar called via MCP), so the wikilink graph stays
intact. Every handler is wrapped so one bad action cannot abort the rest of
the batch — integrity-first: apply what we safely can, record the rest.

The boundary validators in ``_schemas`` guarantee each action is
well-formed before it reaches here; this module is the deterministic apply
step, and the post-write safety net in ``scholar.review`` is the backstop.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import yaml

from . import _validation, library_tools

if TYPE_CHECKING:
    from ._schemas import (
        ScholarAction,
        ScholarArchiveEntry,
        ScholarDecisions,
        ScholarDeprecateEntry,
        ScholarMergeEntries,
        ScholarMoveEntry,
        ScholarUpdateEntry,
        ScholarWriteEntry,
    )

log = logging.getLogger("agent_mem_daemon.scholar_executor")


@dataclass
class ExecResult:
    """Outcome of applying a ``ScholarDecisions`` batch.

    ``counts`` is rolled into the run's ``decisions`` audit dict; ``notes``
    is a flat list of human-readable per-action lines for the daemon log.
    """

    counts: Dict[str, int] = field(default_factory=dict[str, int])
    notes: List[str] = field(default_factory=list[str])

    def _bump(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def record_ok(self, action: str, note: str) -> None:
        self._bump("actions_applied")
        self._bump(action)
        self.notes.append(note)

    def record_failure(self, action: str, note: str) -> None:
        self._bump("actions_failed")
        self.notes.append(f"FAILED {action}: {note}")


# ── Frontmatter (re)serialisation ────────────────────────────────────────


def _split_body(text: str) -> Tuple[Dict[str, Any], str]:
    """Return ``(frontmatter_mapping, body_after_frontmatter)``. The mapping
    is ``{}`` when there is no parseable frontmatter, in which case the whole
    text is treated as body."""
    fm = _validation.parse_frontmatter(text)
    body = _validation.strip_frontmatter(text)
    return fm, body


def _reserialise(fm: Dict[str, Any], body: str) -> str:
    """Re-join a frontmatter mapping and a body into a full entry document."""
    dumped = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False).rstrip()
    return f"---\n{dumped}\n---\n{body}"


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ── index.md row maintenance ─────────────────────────────────────────────

_INDEX_HEADER = (
    "# Knowledge Index\n\n"
    "| Article | Scope | Status | Conf | Summary | Applies-when (first) | "
    "Compiled From | Updated |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _wikilink_for(rel_path: str) -> str:
    """The canonical entry wikilink target for a knowledge-relative path
    (strip a trailing ``.md``)."""
    return rel_path[:-3] if rel_path.endswith(".md") else rel_path


def _summary_from(fm: Dict[str, Any], body: str) -> str:
    """Best-effort one-line summary for the index row: the frontmatter
    ``summary`` if present, else the entry title, else the first non-heading
    prose line of the body."""
    for key in ("summary", "title"):
        val = str(fm.get(key, "") or "").strip()
        if val:
            return val.replace("|", "/").replace("\n", " ")
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped.replace("|", "/")[:160]
    return ""


def _applies_when_first(fm: Dict[str, Any]) -> str:
    """First line of the (possibly multi-line) ``applies-when`` field."""
    raw: object = fm.get("applies-when", "")
    if isinstance(raw, list):
        items = cast(List[object], raw)
        raw = items[0] if items else ""
    text = str(raw or "").strip()
    return text.splitlines()[0].strip().replace("|", "/") if text else ""


def _index_row(rel_path: str, fm: Dict[str, Any], body: str, *, session_id: str) -> str:
    """Build one catalog row for an entry from its frontmatter + body."""
    sources: object = fm.get("sources", "")
    if isinstance(sources, list):
        items = cast(List[object], sources)
        compiled = ", ".join(str(s) for s in items) if items else ""
    else:
        compiled = str(sources or "")
    if not compiled and session_id:
        compiled = f"session:{session_id}"
    cells = [
        f"[[{_wikilink_for(rel_path)}]]",
        str(fm.get("scope", "") or ""),
        str(fm.get("status", "") or ""),
        str(fm.get("confidence", "") or ""),
        _summary_from(fm, body),
        _applies_when_first(fm),
        compiled.replace("|", "/"),
        str(fm.get("updated", "") or ""),
    ]
    return "| " + " | ".join(c.replace("\n", " ") for c in cells) + " |"


def _row_targets_path(line: str, rel_path: str) -> bool:
    """True iff ``line`` is a table row whose wikilink is ``rel_path``."""
    if not line.lstrip().startswith("|"):
        return False
    target = _wikilink_for(rel_path)
    return f"[[{target}]]" in line or f"[[{target}|" in line or f"[[{rel_path}]]" in line


def _read_index(knowledge_dir: Path) -> str:
    idx = knowledge_dir / "index.md"
    try:
        return idx.read_text(encoding="utf-8")
    except OSError:
        return ""


def _upsert_index_row(knowledge_dir: Path, rel_path: str, row: str) -> None:
    """Insert or replace the catalog row for ``rel_path`` in index.md.

    Creates index.md (with the canonical header) when missing. Replaces an
    existing row for the same entry in place; otherwise appends after the
    last table row."""
    text = _read_index(knowledge_dir)
    if not text.strip():
        _atomic_write(knowledge_dir / "index.md", _INDEX_HEADER + row + "\n")
        return
    out: List[str] = []
    replaced = False
    last_row_idx = -1
    for line in text.splitlines():
        if _row_targets_path(line, rel_path):
            out.append(row)
            replaced = True
        else:
            out.append(line)
            if line.lstrip().startswith("|"):
                last_row_idx = len(out) - 1
    if not replaced:
        insert_at = last_row_idx + 1 if last_row_idx >= 0 else len(out)
        out.insert(insert_at, row)
    _atomic_write(knowledge_dir / "index.md", "\n".join(out) + "\n")


def _remove_index_row(knowledge_dir: Path, rel_path: str) -> None:
    """Drop the catalog row for ``rel_path`` from index.md (no-op if absent)."""
    text = _read_index(knowledge_dir)
    if not text.strip():
        return
    kept = [ln for ln in text.splitlines() if not _row_targets_path(ln, rel_path)]
    _atomic_write(knowledge_dir / "index.md", "\n".join(kept) + "\n")


# ── log.md audit append ──────────────────────────────────────────────────


def _append_log(
    knowledge_dir: Path,
    *,
    action: str,
    target: str,
    session_id: str,
    note: str,
    now: datetime,
) -> None:
    """Append one audit block to knowledge/log.md (best-effort)."""
    block = (
        f"\n## [{now.isoformat(timespec='seconds')}] {action} | {target}\n"
        f"- Source: session:{session_id or 'batch'}\n"
        f"- {note}\n"
    )
    log_path = knowledge_dir / "log.md"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, block.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError as e:
        log.warning("scholar_executor: could not append to log.md: %s", e)


# ── Per-action handlers ──────────────────────────────────────────────────


def _apply_write_like(
    knowledge_dir: Path,
    rel_path: str,
    body: str,
    *,
    action: str,
    session_id: str,
    now: datetime,
    result: ExecResult,
) -> None:
    """Shared body for write_entry / update_entry (write the body verbatim,
    sync the index row, append the log)."""
    abs_path = (knowledge_dir / rel_path).resolve()
    _atomic_write(abs_path, body)
    fm, fm_body = _split_body(body)
    row = _index_row(rel_path, fm, fm_body, session_id=session_id)
    _upsert_index_row(knowledge_dir, rel_path, row)
    _append_log(
        knowledge_dir,
        action=action,
        target=rel_path,
        session_id=session_id,
        note=f"wrote {len(body)} chars",
        now=now,
    )
    result.record_ok(action, f"{action}: {rel_path}")


def _do_write(
    kd: Path, a: "ScholarWriteEntry", *, session_id: str, now: datetime, result: ExecResult
) -> None:
    _apply_write_like(
        kd, a.path, a.body, action="write_entry", session_id=session_id, now=now, result=result
    )


def _do_update(
    kd: Path, a: "ScholarUpdateEntry", *, session_id: str, now: datetime, result: ExecResult
) -> None:
    _apply_write_like(
        kd,
        a.path,
        a.new_body,
        action="update_entry",
        session_id=session_id,
        now=now,
        result=result,
    )


def _do_merge(
    kd: Path, a: "ScholarMergeEntries", *, session_id: str, now: datetime, result: ExecResult
) -> None:
    """Write the merged target, then archive every distinct source path."""
    _atomic_write((kd / a.target_path).resolve(), a.target_body)
    fm, fm_body = _split_body(a.target_body)
    _upsert_index_row(
        kd, a.target_path, _index_row(a.target_path, fm, fm_body, session_id=session_id)
    )
    archived: List[str] = []
    for src in a.source_paths:
        if src == a.target_path:
            continue  # target overwritten in place; don't archive it
        if _archive_path(kd, src, now=now):
            _remove_index_row(kd, src)
            archived.append(src)
    _append_log(
        kd,
        action="merge_entries",
        target=a.target_path,
        session_id=session_id,
        note=f"merged {len(archived)} source(s) → {a.target_path}",
        now=now,
    )
    result.record_ok("merge_entries", f"merge_entries: {a.target_path} (archived {archived})")


def _do_move(
    kd: Path, a: "ScholarMoveEntry", *, session_id: str, now: datetime, result: ExecResult
) -> None:
    """Atomic move + inbound-wikilink rewrite via library_tools, then move
    the index row to the new path."""
    to_folder = a.to_path.rsplit("/", 1)[0] if "/" in a.to_path else ""
    response = library_tools.move_entries(
        kd.resolve(), {"to_folder": to_folder, "files": [a.from_path]}
    )
    text = library_tools.unwrap_text_response(response)
    if text.startswith("(move_entries error)"):
        result.record_failure("move_entry", text)
        return
    _remove_index_row(kd, a.from_path)
    moved = (kd / a.to_path).resolve()
    fm, body = _split_body(_safe_read(moved))
    if fm:
        _upsert_index_row(kd, a.to_path, _index_row(a.to_path, fm, body, session_id=session_id))
    _append_log(
        kd,
        action="move_entry",
        target=a.to_path,
        session_id=session_id,
        note=f"{a.from_path} → {a.to_path}",
        now=now,
    )
    result.record_ok("move_entry", f"move_entry: {a.from_path} → {a.to_path}")


def _do_archive(
    kd: Path, a: "ScholarArchiveEntry", *, session_id: str, now: datetime, result: ExecResult
) -> None:
    if not _archive_path(kd, a.path, now=now):
        result.record_failure("archive_entry", f"could not archive {a.path}")
        return
    _remove_index_row(kd, a.path)
    _append_log(
        kd,
        action="archive_entry",
        target=a.path,
        session_id=session_id,
        note=f"archived {a.path} → _archive/{a.path}",
        now=now,
    )
    result.record_ok("archive_entry", f"archive_entry: {a.path}")


def _do_deprecate(
    kd: Path, a: "ScholarDeprecateEntry", *, session_id: str, now: datetime, result: ExecResult
) -> None:
    """Mark the entry deprecated in place: set ``status: deprecated`` +
    ``superseded_by``, insert a banner after the first heading, keep the file
    (so inbound wikilinks still resolve)."""
    abs_path = (kd / a.path).resolve()
    text = _safe_read(abs_path)
    if not text:
        result.record_failure("deprecate_entry", f"could not read {a.path}")
        return
    fm, body = _split_body(text)
    if not fm:
        result.record_failure("deprecate_entry", f"{a.path} has no parseable frontmatter")
        return
    fm["status"] = "deprecated"
    fm["superseded_by"] = a.superseded_by
    banner = (
        f"> **Superseded by [[{_wikilink_for(a.superseded_by)}]] as of "
        f"{now.date().isoformat()}**. Kept for historical reference."
    )
    new_body = _insert_after_first_heading(body, banner)
    _atomic_write(abs_path, _reserialise(fm, new_body))
    _upsert_index_row(kd, a.path, _index_row(a.path, fm, new_body, session_id=session_id))
    _append_log(
        kd,
        action="deprecate_entry",
        target=a.path,
        session_id=session_id,
        note=f"deprecated {a.path}; superseded by {a.superseded_by}",
        now=now,
    )
    result.record_ok("deprecate_entry", f"deprecate_entry: {a.path} → {a.superseded_by}")


# ── Small shared helpers ─────────────────────────────────────────────────


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _insert_after_first_heading(body: str, banner: str) -> str:
    """Insert ``banner`` (as its own paragraph) immediately after the first
    ``# Heading`` line, or at the top if there is no heading."""
    if banner in body:
        return body
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            lines.insert(idx + 1, "")
            lines.insert(idx + 2, banner)
            return "\n".join(lines) + ("\n" if body.endswith("\n") else "")
    return f"{banner}\n\n{body}"


def _archive_path(knowledge_dir: Path, rel_path: str, *, now: datetime) -> bool:
    """Move ``rel_path`` into ``_archive/`` preserving structure, stamping
    ``status: stale`` + ``archived: <today>``. Best-effort: returns False on
    any failure, leaving the source untouched."""
    src = (knowledge_dir / rel_path).resolve()
    text = _safe_read(src)
    if not text:
        return False
    dest = (knowledge_dir / "_archive" / rel_path).resolve()
    fm, body = _split_body(text)
    if fm:
        fm["status"] = "stale"
        fm["archived"] = now.date().isoformat()
        new_text = _reserialise(fm, body)
    else:
        new_text = text
    try:
        _atomic_write(dest, new_text)
        src.unlink()
    except OSError as e:
        log.warning("scholar_executor: archive of %s failed: %s", rel_path, e)
        return False
    return True


# ── Dispatch ─────────────────────────────────────────────────────────────


def _dispatch_one(
    kd: Path,
    action: "ScholarAction",
    *,
    session_id: str,
    now: datetime,
    result: ExecResult,
) -> None:
    """Route one validated action to its handler. Isinstance dispatch (not a
    dict) so each handler keeps its precise parameter type under strict
    typing; the import-time union guarantees exhaustiveness."""
    from ._schemas import (  # noqa: PLC0415 — runtime isinstance needs the classes
        ScholarArchiveEntry,
        ScholarMergeEntries,
        ScholarMoveEntry,
        ScholarUpdateEntry,
        ScholarWriteEntry,
    )

    if isinstance(action, ScholarWriteEntry):
        _do_write(kd, action, session_id=session_id, now=now, result=result)
    elif isinstance(action, ScholarUpdateEntry):
        _do_update(kd, action, session_id=session_id, now=now, result=result)
    elif isinstance(action, ScholarMergeEntries):
        _do_merge(kd, action, session_id=session_id, now=now, result=result)
    elif isinstance(action, ScholarMoveEntry):
        _do_move(kd, action, session_id=session_id, now=now, result=result)
    elif isinstance(action, ScholarArchiveEntry):
        _do_archive(kd, action, session_id=session_id, now=now, result=result)
    else:
        # The only remaining union member is ScholarDeprecateEntry; pyright
        # narrows it here without a redundant isinstance check.
        _do_deprecate(kd, action, session_id=session_id, now=now, result=result)


def apply_decisions(
    decisions: "ScholarDecisions",
    knowledge_dir: Path,
    *,
    session_id: str = "",
    now: datetime | None = None,
) -> ExecResult:
    """Apply a validated ``ScholarDecisions`` to disk, in list order.

    Each action is applied by its handler; one failing action is recorded
    and skipped (integrity-first — never abort the whole batch). Returns an
    :class:`ExecResult` with counters for the audit row and human-readable
    notes for the log.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    result = ExecResult()
    for action in decisions.actions:
        try:
            _dispatch_one(knowledge_dir, action, session_id=session_id, now=now, result=result)
        except Exception as e:  # noqa: BLE001 — one bad action must not abort the batch
            log.exception("scholar_executor: action %s raised", action.action)
            result.record_failure(str(action.action), str(e))
    return result
