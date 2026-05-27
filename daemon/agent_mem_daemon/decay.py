"""Memory decay engine (Slice 1 of the LTD design).

Implements deterministic, no-LLM memory decay against the spec in
``[[projects/agent-mem/concepts/forgetting-ltd-decay-design]]``:

- **last_surfaced bookkeeping.** The daemon stamps an entry's
  frontmatter each time the entry's wikilink lands in priming output
  (the per-session sent-cache is the trigger). This is the "the agent
  has had a chance to see this" signal.

- **Periodic sweep.** Walks ``~/.agent-mem/knowledge/`` and archives
  entries whose ``last_activity`` (max of created, updated,
  last_reinforced, last_surfaced) is older than ``DECAY_AGE_DAYS``,
  whose ``reinforced`` counter is below ``DECAY_REINFORCED_FLOOR``,
  and which aren't ``arousal_pinned: true``. Archived = moved to
  ``_archive/<orig-path>`` with ``status: stale`` and an
  ``archived: <today>`` frontmatter field.

- **Opportunistic trigger.** No dedicated thread. The Scholar's
  post-batch flow and the priming RPC each call
  :func:`maybe_run_sweep`; the function self-skips unless
  ``SWEEP_MIN_INTERVAL_HOURS`` has elapsed since the last successful
  sweep. Sweep state lives at ``~/.agent-mem/sweep-state.json``.

The whole module is failure-tolerant: every entry is archived
independently, every disk write is best-effort with logged exceptions,
and the public entry points (:func:`stamp_last_surfaced`,
:func:`run_sweep`, :func:`maybe_run_sweep`) never raise. The daemon
must never crash because a decay pass hit a malformed file.

Concurrency: ``_SWEEP_LOCK`` guards against concurrent sweeps from the
two trigger sites; frontmatter writes use temp+rename atomicity so
overlapping stamps from different threads can't corrupt an entry.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import yaml

from .paths import home as _agent_mem_home

log = logging.getLogger("agent_mem_daemon.decay")


# ── Tunables (match the design's spec'd defaults) ────────────────────


# Entries idle for this many days are eligible for archival.
DECAY_AGE_DAYS = 30

# Entries with ``reinforced >= this`` survive regardless of age. The
# design's "reinforcement extends life" intuition — once the user has
# reasserted an entry, it's not noise.
DECAY_REINFORCED_FLOOR = 2

# Minimum interval between sweeps. The Scholar's post-batch flow and
# the priming RPC both call ``maybe_run_sweep`` — without this guard
# we'd sweep on every batch and every hook turn.
SWEEP_MIN_INTERVAL_HOURS = 24


# ── Module-level state ───────────────────────────────────────────────


_SWEEP_LOCK = threading.Lock()


# Matches the same shape ``scholar_prompt`` uses for frontmatter
# extraction so a single regex govern both modules' read paths.
_FRONTMATTER_HEAD_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _sweep_state_path() -> Path:
    return _agent_mem_home() / "sweep-state.json"


def _knowledge_dir() -> Path:
    return _agent_mem_home() / "knowledge"


def _today_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).date().isoformat()


# ── Frontmatter mutation — last_surfaced bookkeeping ─────────────────


def _read_and_parse_frontmatter(
    entry_path: Path,
) -> Optional[tuple[Dict[str, Any], str, str]]:
    """Return ``(frontmatter_dict, fm_block_str, body)`` or ``None``
    on any failure (unreadable file, no frontmatter fence, malformed
    YAML, top-level not a mapping). Permissive — caller decides what
    to do with a ``None``."""
    try:
        text = entry_path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _FRONTMATTER_HEAD_RE.match(text)
    if not m:
        return None
    try:
        loaded: object = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    fm = cast(Dict[str, Any], loaded)
    body = text[m.end() :]
    return fm, m.group(1), body


def _atomic_write_entry(entry_path: Path, fm: Dict[str, Any], body: str) -> bool:
    """Serialise ``fm`` back to YAML and rewrite the entry atomically.
    Returns True on success, False on any I/O / YAML failure."""
    try:
        new_fm = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False).rstrip()
    except yaml.YAMLError:
        return False
    new_text = f"---\n{new_fm}\n---\n{body}"
    tmp = entry_path.with_suffix(entry_path.suffix + ".tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, entry_path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    return True


def stamp_last_surfaced(
    entry_path: Path,
    *,
    today_iso: Optional[str] = None,
) -> bool:
    """Set ``last_surfaced`` to today's date on the entry's frontmatter.

    Idempotent within a day — if the field is already today's value,
    skip the write. Never raises.

    Args:
        entry_path: absolute path to the entry's ``.md`` file.
        today_iso: ISO date string to write (default: UTC today).
            Tests pin this for deterministic assertions.

    Returns:
        True if the file was updated (or already-current), False on any
        failure path (file missing, bad frontmatter, write failed).
    """
    iso = today_iso or _today_iso()
    parsed = _read_and_parse_frontmatter(entry_path)
    if parsed is None:
        return False
    fm, _raw_fm, body = parsed
    if fm.get("last_surfaced") == iso:
        return True  # No-op: already stamped today.
    fm["last_surfaced"] = iso
    return _atomic_write_entry(entry_path, fm, body)


# ── Sweep state I/O ──────────────────────────────────────────────────


def read_last_sweep_at() -> Optional[datetime]:
    """Return the UTC timestamp of the last successful sweep, or None
    if never run (or state file unreadable/corrupt — same semantics:
    "we don't know when, so let's run one")."""
    p = _sweep_state_path()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data: object = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data_dict = cast(Dict[str, Any], data)
    when = data_dict.get("last_sweep_at")
    if not isinstance(when, str):
        return None
    try:
        # fromisoformat handles the trailing 'Z' on 3.11+.
        return datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError:
        return None


def write_last_sweep_at(when: datetime) -> None:
    """Persist ``when`` to the sweep-state file. Atomic write."""
    p = _sweep_state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    payload = json.dumps({"last_sweep_at": when.astimezone(timezone.utc).isoformat()})
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


# ── Sweep eligibility ────────────────────────────────────────────────


def _parse_iso_date(value: object) -> Optional[date]:
    """Parse YYYY-MM-DD strings out of frontmatter. Returns None on
    anything that doesn't parse cleanly."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Handle both bare dates and ISO timestamps.
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _reinforced_count(fm: Dict[str, Any]) -> int:
    raw = fm.get("reinforced")
    if raw is None:
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _is_arousal_pinned(fm: Dict[str, Any]) -> bool:
    """Identity/safety-critical entries set ``arousal_pin: true``. Slice
    1 just respects the field if present — Scholar gains the ability
    to set it in a later slice."""
    return bool(fm.get("arousal_pin")) or bool(fm.get("arousal_pinned"))


def _last_activity_date(fm: Dict[str, Any]) -> Optional[date]:
    """Most-recent activity timestamp across the four date fields
    decay considers. Returns None if no field parses (treat as
    'unknown age' — caller's choice whether to archive or skip)."""
    candidates = [
        _parse_iso_date(fm.get("last_surfaced")),
        _parse_iso_date(fm.get("last_reinforced")),
        _parse_iso_date(fm.get("updated")),
        _parse_iso_date(fm.get("created")),
    ]
    valid = [d for d in candidates if d is not None]
    if not valid:
        return None
    return max(valid)


def _is_eligible_for_archive(
    fm: Dict[str, Any],
    *,
    today: date,
    max_age_days: int = DECAY_AGE_DAYS,
    reinforced_floor: int = DECAY_REINFORCED_FLOOR,
) -> bool:
    """Apply the three-condition archival rule from the design.

    All three must hold:
      - age (today - last_activity) > max_age_days
      - reinforced < reinforced_floor
      - not arousal_pinned

    Already-archived entries (``status: stale``) are not eligible — the
    sweep doesn't churn the archive directory. Returns False on any
    indeterminate state (no date fields → conservative keep)."""
    if str(fm.get("status", "")).strip().lower() == "stale":
        return False
    if _is_arousal_pinned(fm):
        return False
    if _reinforced_count(fm) >= reinforced_floor:
        return False
    last_activity = _last_activity_date(fm)
    if last_activity is None:
        # No timestamps — conservatively keep. The Scholar/Librarian
        # should fix the entry's frontmatter; we don't archive on
        # missing data.
        return False
    age_days = (today - last_activity).days
    return age_days > max_age_days


# ── Archival ─────────────────────────────────────────────────────────


def _archive_destination(entry_path: Path, knowledge_dir: Path) -> Optional[Path]:
    """Map ``knowledge/<rel>`` -> ``knowledge/_archive/<rel>``. None
    if the entry isn't under the knowledge dir (defensive)."""
    try:
        rel = entry_path.resolve().relative_to(knowledge_dir.resolve())
    except ValueError:
        return None
    return (knowledge_dir / "_archive" / rel).resolve()


def _archive_entry(
    entry_path: Path,
    knowledge_dir: Path,
    fm: Dict[str, Any],
    body: str,
    *,
    today_iso: str,
) -> bool:
    """Move ``entry_path`` into ``_archive/`` preserving subdirectory
    structure, and stamp ``status: stale`` + ``archived: <today>`` in
    the archived file's frontmatter.

    Best-effort: returns False on any failure (no destination resolved,
    parent dir creation failed, write or move failed). On failure the
    original entry is left untouched."""
    dest = _archive_destination(entry_path, knowledge_dir)
    if dest is None:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    # Stamp the archival metadata into the moved entry's frontmatter
    # before the move so the file at its final path is internally
    # consistent.
    fm["status"] = "stale"
    fm["archived"] = today_iso
    try:
        new_fm = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False).rstrip()
    except yaml.YAMLError:
        return False
    new_text = f"---\n{new_fm}\n---\n{body}"
    # Write the new content at the destination, then delete the source.
    # Move-then-modify would leave a stale `archived:` value pointing
    # at whatever the file said before; this ordering is cleaner.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, dest)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    try:
        entry_path.unlink()
    except OSError:
        # Destination is in place but source removal failed. Log; the
        # next sweep will skip the source (now `status: stale` per
        # the destination's content) once the orphan is cleaned up.
        log.warning(
            "decay: archived %s -> %s but failed to remove source",
            entry_path,
            dest,
        )
        return True
    return True


# ── Public sweep API ─────────────────────────────────────────────────


@dataclass
class SweepResult:
    """Summary of one sweep pass — what got moved, what got skipped,
    what failed. Returned by :func:`run_sweep` and logged at INFO."""

    archived: int = 0
    kept: int = 0
    errored: int = 0
    archived_paths: List[str] = field(default_factory=lambda: [])


def _iter_entries(knowledge_dir: Path) -> list[Path]:
    """All markdown entries the sweep considers — everything under
    ``knowledge_dir`` except ``_archive/`` itself and the top-level
    ``index.md`` / ``log.md`` catalogs (those are daemon-maintained,
    not user content)."""
    if not knowledge_dir.exists():
        return []
    out: list[Path] = []
    skip_names = {"index.md", "log.md"}
    for p in knowledge_dir.rglob("*.md"):
        try:
            rel = p.resolve().relative_to(knowledge_dir.resolve())
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "_archive":
            continue
        if rel.parent == Path() and rel.name in skip_names:
            continue
        out.append(p)
    return out


def run_sweep(
    knowledge_dir: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> SweepResult:
    """Walk the library and archive eligible entries. Always updates
    the sweep-state file when complete so the cooldown takes effect.

    Args:
      knowledge_dir: override for the knowledge root (tests pin this).
      now: override for the date check (tests pin this).

    Returns:
      :class:`SweepResult` with counts and the list of archived paths.

    Never raises. Each entry is independent; one bad entry doesn't
    abort the sweep.
    """
    kdir = knowledge_dir or _knowledge_dir()
    when = now or datetime.now(timezone.utc)
    today = when.date()
    today_iso = today.isoformat()
    result = SweepResult()

    if not kdir.exists():
        log.debug("decay.run_sweep: knowledge dir %s missing — nothing to sweep", kdir)
        write_last_sweep_at(when)
        return result

    for entry_path in _iter_entries(kdir):
        try:
            parsed = _read_and_parse_frontmatter(entry_path)
            if parsed is None:
                result.errored += 1
                continue
            fm, _raw, body = parsed
            if not _is_eligible_for_archive(fm, today=today):
                result.kept += 1
                continue
            ok = _archive_entry(entry_path, kdir, fm, body, today_iso=today_iso)
            if ok:
                result.archived += 1
                rel = ""
                try:
                    rel = str(entry_path.relative_to(kdir))
                except ValueError:
                    rel = str(entry_path)
                result.archived_paths.append(rel)
            else:
                result.errored += 1
        except Exception:
            log.exception("decay.run_sweep: unexpected failure on %s", entry_path)
            result.errored += 1

    write_last_sweep_at(when)
    log.info(
        "decay.sweep: archived=%d kept=%d errored=%d",
        result.archived,
        result.kept,
        result.errored,
    )
    return result


def should_sweep(
    *,
    now: Optional[datetime] = None,
    min_interval_hours: float = SWEEP_MIN_INTERVAL_HOURS,
) -> bool:
    """True if the cooldown has elapsed since the last sweep (or no
    sweep has ever run)."""
    last = read_last_sweep_at()
    if last is None:
        return True
    when = now or datetime.now(timezone.utc)
    elapsed_hours = (when - last).total_seconds() / 3600.0
    return elapsed_hours >= min_interval_hours


def maybe_run_sweep(
    knowledge_dir: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> Optional[SweepResult]:
    """Run a sweep iff the cooldown has elapsed AND no other sweep is
    in flight. Returns the :class:`SweepResult` if a sweep ran,
    ``None`` if the trigger was skipped.

    The lock makes the Scholar batch and priming RPC paths safe to
    both call this without coordinating — second concurrent caller
    short-circuits."""
    if not _SWEEP_LOCK.acquire(blocking=False):
        return None
    try:
        if not should_sweep(now=now):
            return None
        return run_sweep(knowledge_dir=knowledge_dir, now=now)
    finally:
        _SWEEP_LOCK.release()


__all__ = [
    "DECAY_AGE_DAYS",
    "DECAY_REINFORCED_FLOOR",
    "SWEEP_MIN_INTERVAL_HOURS",
    "SweepResult",
    "maybe_run_sweep",
    "read_last_sweep_at",
    "run_sweep",
    "should_sweep",
    "stamp_last_surfaced",
    "write_last_sweep_at",
]
