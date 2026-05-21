"""Pending-nudges file reader + per-session budget enforcement.

The Scholar (in the daemon) writes ratified interrupts to
``~/.agent-mem/pending-nudges.md`` (path resolved via ``AGENT_MEM_HOME``).
The UserPromptSubmit hook calls :func:`take_nudges` to:

1. Read and parse the file.
2. Pick up to ``per_turn_budget`` nudges respecting the
   ``per_session_budget`` (1/turn, 3/session per PLAN §5).
3. Atomically clear the file (move to a ``.consumed`` sibling — keeps
   the most recent consumption visible for debugging).
4. Update the per-session budget counter file.

The hook itself is a thin shell around this module so the logic is
testable without spawning a subprocess. Pure file I/O — no LLM calls —
so a healthy invocation lands well under the < 50 ms budget.

File format (matches ``daemon/agent_mem_daemon/scholar_prompt.py`` —
single source of truth lives in that module's docstring)::

    ---
    id: <short uuid>
    created: <iso8601>
    lesson: <path/to/entry>
    ---
    <one-paragraph nudge text>

Blocks separated by another ``---``. The daemon appends; we read-and-clear.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# PLAN §5 — pinned per-turn / per-session budgets.
DEFAULT_PER_TURN_BUDGET = 1
DEFAULT_PER_SESSION_BUDGET = 3


# ── Path resolution ───────────────────────────────────────────────────


def _agent_mem_home() -> Path:
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
    # characters from synthetic test inputs.
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
    """Parse pending-nudges.md content into ordered Nudge objects.

    Mirrors ``daemon.scholar_prompt.parse_nudges_file`` byte-for-byte
    semantically — we duplicate the implementation here because the
    hook can't depend on the daemon package (the two live in different
    pyproject roots and we keep the hook side dependency-free).
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


# ── Budget bookkeeping ────────────────────────────────────────────────


def _load_budget(session_id: str) -> Dict[str, int]:
    path = budget_state_path(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                "consumed": int(data.get("consumed", 0)),
            }
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return {"consumed": 0}


def _save_budget(session_id: str, consumed: int) -> None:
    """Atomically persist the per-session counter via temp + rename."""
    path = budget_state_path(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    data = {"consumed": int(consumed), "updated": time.time()}
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_name, path)
        except Exception:
            # Best effort: try to clean up the tmp file if rename failed.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError:
        # Persistence failure is non-fatal. Worst case: same session
        # gets one more nudge than budget. We log nothing — hook stderr
        # is captured by Claude Code and noise would confuse the user.
        return


# ── Nudge file consumption ────────────────────────────────────────────


def _read_and_clear_nudges_file(path: Path) -> str:
    """Read the nudges file and move it aside in one shot.

    We rename the file to ``<path>.consumed`` (with a small rotation:
    only the most-recent consumed file is kept; older ones overwritten).
    Renaming is the cheapest atomic clear available — no need to
    truncate-then-write.

    Returns the body as a string, or empty string on any failure.
    """
    try:
        body = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    if not body:
        # File existed but is empty — still try to clean up.
        try:
            path.unlink()
        except OSError:
            pass
        return ""
    consumed_path = path.with_suffix(path.suffix + ".consumed")
    try:
        # ``os.replace`` is atomic on POSIX. On Windows it overwrites
        # any existing target, which is what we want here too.
        os.replace(path, consumed_path)
    except OSError:
        # If we can't move it, try to truncate instead so we don't
        # re-emit the same nudges next turn.
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.truncate(0)
        except OSError:
            pass
    return body


def take_nudges(
    session_id: str,
    *,
    per_turn_budget: int = DEFAULT_PER_TURN_BUDGET,
    per_session_budget: int = DEFAULT_PER_SESSION_BUDGET,
    now: Optional[float] = None,
    _nudges_path: Optional[Path] = None,
    _budget_state_path: Optional[Path] = None,
) -> Tuple[List[Nudge], int]:
    """Pop up to ``per_turn_budget`` nudges for this turn.

    Honours the per-session ceiling: even if 100 nudges are queued, a
    session that has already shown 3 returns ``([], 3)``.

    Args:
        session_id: derived from the hook's stdin payload.
        per_turn_budget: max nudges this single turn may emit.
        per_session_budget: cumulative cap across the whole session.
        now: clock override for tests.
        _nudges_path / _budget_state_path: test seams; production code
            resolves these via the env-var-aware helpers above.

    Returns:
        (selected_nudges, consumed_after) — ``consumed_after`` is the
        session's running total post-emission, useful for callers that
        want to inject a "review batch" hint.
    """
    if now is None:
        now = time.time()
    nudges_path = _nudges_path if _nudges_path is not None else pending_nudges_path()

    # We always read + clear the file, even if the per-session budget is
    # already exhausted — otherwise nudges would accumulate forever
    # waiting for a session that has already declined them all. This
    # matches PLAN §5: the budget is a per-session ceiling, not a queue.
    body = _read_and_clear_nudges_file(nudges_path)
    queued = parse_nudges(body)

    # Load consumed counter. We compute the budget path lazily so tests
    # can inject one without having to set AGENT_MEM_HOME globally.
    if _budget_state_path is not None:
        try:
            data = json.loads(_budget_state_path.read_text(encoding="utf-8"))
            consumed = int(data.get("consumed", 0)) if isinstance(data, dict) else 0
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
            consumed = 0
    else:
        consumed = _load_budget(session_id)["consumed"]

    remaining_session = max(0, per_session_budget - consumed)
    take = min(per_turn_budget, remaining_session, len(queued))
    selected = queued[:take]
    # Anything beyond ``take`` is dropped — we already cleared the file
    # so it doesn't roll over. PLAN §5: "without this budget, this
    # becomes Clippy." The Scholar should also be aggressively vetoing,
    # so the overflow case ought to be rare.
    new_consumed = consumed + len(selected)

    if selected:
        if _budget_state_path is not None:
            # Test path — write directly.
            try:
                _budget_state_path.parent.mkdir(parents=True, exist_ok=True)
                _budget_state_path.write_text(
                    json.dumps({"consumed": new_consumed, "updated": now}),
                    encoding="utf-8",
                )
            except OSError:
                pass
        else:
            _save_budget(session_id, new_consumed)

    return selected, new_consumed


# ── Context rendering ─────────────────────────────────────────────────


def render_context(nudges: List[Nudge]) -> str:
    """Format selected nudges as the ``additionalContext`` string Claude
    Code sees on the next user turn.

    Style follows SCHOLAR_PROMPT.md §2 PENDING NUDGES guidance + the
    PLAN §5 framing: factual, present-tense, "the user has not been
    asked." Apply if relevant.
    """
    if not nudges:
        return ""
    lines = [
        f"Memory has {len(nudges)} relevant lesson(s). "
        "Apply if relevant. The user has not been asked."
    ]
    for n in nudges:
        # One bullet per nudge: lesson path then the text.
        ref = f"[[{n.lesson}]]" if n.lesson else "(unknown lesson)"
        lines.append(f"- {ref}: {n.text}")
    return "\n".join(lines)
