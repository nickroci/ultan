"""Prompt assembly + response parsing for the Librarian.

Pure functions. No I/O happens at import time. The daemon orchestrator
(``librarian.scan``) wires these together with the typed agent shim
(``typed_agent.run_typed``) and the audit machinery in ``runs.py``.

The Librarian is the *active organiser* in the new architecture. It
receives:

  1. The conversation buffer (rolling-buffer snapshot).
  2. A snapshot of the library's current state:
       - directory tree
       - root README excerpt
       - index.md content (truncated)
       - top-level folder READMEs
…and emits a typed `LibrarianProposal`: a list of `ProposedAction` items.
The Librarian has read-only research tools (``read_entry``,
``grep_library``, ``bm25_search`` for lexical match, ``embedding_search``
for semantic match) wired in-process by ``_agent_research`` that it fans
out when checking for duplicates / contradictions. The typed shim returns
the validated typed object directly — there is no JSON to scrape.

The Scholar is the gatekeeper that approves/vetoes each proposal —
the Librarian never writes to disk.

See ``docs/LIBRARIAN_PROMPT.md`` for the full spec.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import yaml
from aliases import session_bucket  # type: ignore[import-not-found]

from . import repair_queue
from ._schemas import (
    describe_action_types_markdown,
    describe_librarian_response_shape,
)
from .paths import home as _home

log = logging.getLogger("agent_mem_daemon.librarian_prompt")


# Hard ceiling on the library snapshot block: we don't want a sprawling
# corpus to push the Librarian's prompt past Haiku's sensible context
# size. ~3 KB per the cost-discipline brief.
LIBRARY_SNAPSHOT_MAX_CHARS = 3 * 1024


# Hard ceiling on the rolling-buffer block — the single largest, most
# variable part of the Librarian prompt. Feeding the WHOLE session every
# pass blew prompts up to 259K chars (median 71K), driving cost (~$100 /
# 2 days) and OOM-killing the spawned SDK subprocess ("Fatal error in
# message reader: exit code -N"). Re-scanning the whole session is also
# pointless: anything already learned is persisted in the library, so the
# Librarian only needs the most RECENT activity to catch NEW lessons.
#
# Budget is expressed in tokens and converted to a char proxy at ~4
# chars/token. We keep the most-recent turns and drop the OLDEST when
# over budget, always retaining at least the single most recent turn.
# Tune ROLLING_BUFFER_BUDGET_TOKENS to trade recall depth against cost.
ROLLING_BUFFER_BUDGET_TOKENS = 30_000
_CHARS_PER_TOKEN = 4
ROLLING_BUFFER_MAX_CHARS = ROLLING_BUFFER_BUDGET_TOKENS * _CHARS_PER_TOKEN


# ── Shape definitions ─────────────────────────────────────────────────
#
# Buffer snapshots come from ``buffer.BufferStore.snapshot()`` as plain
# dicts; their detailed shape varies (extra keys, optional payload
# fields), so the snapshot-walking functions below accept the broadest
# safe signature, ``Mapping[str, Any]``, and validate each level with
# ``_as_str_map`` / ``isinstance(list, ...)`` at the boundary.


def _as_str_map(obj: object) -> Mapping[str, Any]:
    """Coerce an arbitrary object to a ``Mapping[str, Any]``.

    Returns an empty mapping if ``obj`` is not a dict-like. Used at the
    edges where we read free-form payload blobs out of the buffer
    snapshot — pyright's ``isinstance(x, Mapping)`` narrowing loses key
    types, so the cast happens here once instead of at every call site.
    """
    if not isinstance(obj, dict):
        return {}
    return cast(Mapping[str, Any], obj)


# ── Buffer flattening (unchanged from the previous design) ────────────


_TEXT_KEYS: Tuple[str, ...] = ("text", "prompt", "response", "content", "message", "summary")


def _payload_role(ev: Mapping[str, Any]) -> str:
    """Map an event to a [role] tag for the Librarian's prompt."""
    pl = _as_str_map(ev.get("payload"))
    explicit = pl.get("role")
    if isinstance(explicit, str) and explicit:
        return explicit.lower()
    typ = ev.get("type") or ""
    if typ == "UserPromptSubmit":
        return "user"
    if typ in ("SessionEnd", "SessionStart"):
        return "system"
    return "assistant"


def _payload_text(ev: Mapping[str, Any]) -> str:
    """Pull a string body out of an event payload."""
    pl = _as_str_map(ev.get("payload"))
    if not pl:
        return ""
    for k in _TEXT_KEYS:
        v = pl.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    tool = pl.get("tool") or pl.get("name")
    if isinstance(tool, str):
        args = _as_str_map(pl.get("arguments") or pl.get("args") or pl.get("input"))
        if args:
            keys = ",".join(sorted(args.keys())[:4])
            return f"{tool}({keys})"
        return str(tool)
    return ""


def flatten_buffer(snapshot: Mapping[str, Any]) -> List[Tuple[int, int, str, str, bool]]:
    """Materialise a buffer snapshot as a list of
    ``(turn_id, turn_seq, role, text, user_asserted)``.

    ``turn_id`` is a 1-based monotonic counter across all quotable events
    in all sealed turns. It is recomputed FRESH on every scan, so it is
    NOT stable across scans — as old turns age out of the deque, the same
    physical event's ``turn_id`` shifts down. It is the human-readable line
    label the Librarian quotes in free-text ``reasoning``.

    ``turn_seq`` is the owning turn's STABLE per-session id (assigned by
    the buffer at seal time; see ``buffer.Turn.turn_seq``). It does NOT
    shift across scans, so it is the id the Librarian cites structurally in
    ``cited_turn_seq`` on a ``used_helpfully`` proposal — the fired-helpful
    counter dedups on it (see ``scholar_prompt.apply_fired_helpful_
    counters``). Events in the same turn share a ``turn_seq``. Older
    snapshots / test fixtures without a ``turn_seq`` field yield 0.

    ``user_asserted`` marks ``/ultan`` events — the Librarian treats those
    as user-stated rules and is told to file them rather than veto.

    Stop / SessionEnd marker events are skipped — they carry no dialogue.
    """
    out: List[Tuple[int, int, str, str, bool]] = []
    counter = 0
    raw_turns_obj: object = snapshot.get("turns") or []
    if not isinstance(raw_turns_obj, list):
        return out
    raw_turns = cast(List[Any], raw_turns_obj)
    for turn_obj in raw_turns:
        turn_map = _as_str_map(turn_obj)
        if not turn_map:
            continue
        raw_seq: object = turn_map.get("turn_seq")
        turn_seq = raw_seq if isinstance(raw_seq, int) and not isinstance(raw_seq, bool) else 0
        raw_events_obj: object = turn_map.get("events") or []
        if not isinstance(raw_events_obj, list):
            continue
        raw_events = cast(List[Any], raw_events_obj)
        for ev_obj in raw_events:
            ev_map = _as_str_map(ev_obj)
            if not ev_map:
                continue
            typ = ev_map.get("type")
            if typ in ("Stop", "SessionEnd"):
                continue
            text = _payload_text(ev_map)
            if not text:
                continue
            pl = _as_str_map(ev_map.get("payload"))
            user_asserted = bool(pl.get("user_asserted"))
            counter += 1
            out.append((counter, turn_seq, _payload_role(ev_map), text, user_asserted))
    return out


def _render_turn_line(tid: int, turn_seq: int, role: str, text: str, user_asserted: bool) -> str:
    """Render one ``(turn_id, turn_seq, role, text, user_asserted)`` tuple
    as the single ``[id] (turn_seq=S) [role] [USER-ASSERTED?] <squashed
    text>`` line the Librarian sees. Shared by the formatter and the
    recency-cap char accounting so the cap is measured against the EXACT
    bytes that land in the prompt.

    ``[id]`` is the scan-local label the Librarian quotes in prose;
    ``(turn_seq=S)`` is the STABLE id it must echo in ``cited_turn_seq``
    on a ``used_helpfully`` proposal. ``turn_seq=0`` (unsealed / legacy
    fixtures with no seq) is rendered too so the token is always present
    and parsing stays uniform."""
    squashed = " ".join(text.split())
    prefix = "[USER-ASSERTED] " if user_asserted else ""
    return f"[{tid}] (turn_seq={turn_seq}) [{role}] {prefix}{squashed}"


def cap_buffer_to_recent(
    flat: Sequence[Tuple[int, int, str, str, bool]],
    *,
    max_chars: int = ROLLING_BUFFER_MAX_CHARS,
) -> List[Tuple[int, int, str, str, bool]]:
    """Trim ``flat`` to the most-recent turns that fit within ``max_chars``.

    The rendered ``<rolling_buffer>`` block is the dominant, unbounded
    part of the Librarian prompt; feeding an entire session here is what
    pushed prompts to 259K chars and OOM-killed the SDK subprocess. We
    walk the turns NEWEST-first, accumulating their rendered line length
    (the exact bytes ``format_rolling_buffer`` will emit, newline
    included), and stop once the next-oldest turn would blow the budget.
    Always keeps at least the single most-recent turn even if that one
    line alone exceeds ``max_chars`` — dropping everything would defeat
    the Librarian entirely.

    Returns the kept turns in their original (oldest-first) order so the
    caller can format them unchanged.
    """
    if not flat:
        return []
    kept_rev: List[Tuple[int, int, str, str, bool]] = []
    used = 0
    for tid, turn_seq, role, text, user_asserted in reversed(flat):
        line_len = len(_render_turn_line(tid, turn_seq, role, text, user_asserted))
        # +1 for the newline join cost between lines (the very first kept
        # line has no separator, but over-counting by one is harmless and
        # keeps the accounting a strict upper bound on the joined output).
        cost = line_len + 1
        if kept_rev and used + cost > max_chars:
            break
        kept_rev.append((tid, turn_seq, role, text, user_asserted))
        used += cost
    kept_rev.reverse()
    dropped = len(flat) - len(kept_rev)
    if dropped:
        log.info(
            "librarian buffer truncated to recency budget: kept %d of %d turns "
            "(dropped %d oldest, budget=%d chars)",
            len(kept_rev),
            len(flat),
            dropped,
            max_chars,
        )
    return kept_rev


def format_rolling_buffer(flat: Sequence[Tuple[int, int, str, str, bool]]) -> str:
    """Render the ``(turn_id, turn_seq, role, text, user_asserted)`` list
    as the ``<rolling_buffer>`` body.

    User-asserted turns get a ``[USER-ASSERTED]`` prefix so the
    Librarian knows the user explicitly named the rule (treat with
    higher priority).
    """
    if not flat:
        return "(empty — no turns with quotable text)"
    lines = [
        _render_turn_line(tid, turn_seq, role, text, user_asserted)
        for tid, turn_seq, role, text, user_asserted in flat
    ]
    return "\n".join(lines)


# ── Library snapshot (new in this architecture) ───────────────────────


# Files we never include in the tree listing — they're catalog/audit
# files, not entries.
_TREE_EXCLUDE_NAMES = frozenset({"index.md", "log.md", ".bm25.idx"})


def _tree_listing(knowledge_dir: Path, *, max_lines: int = 80) -> str:
    """Render a compact tree of ``knowledge_dir``.

    Layout:
        global/
          README.md
          tooling/
            README.md
            uv-basics.md
            ...
        projects/
          ...

    Capped at ``max_lines`` to keep the snapshot bounded. Truncation is
    signalled with ``... (N more lines truncated)``.
    """
    if not knowledge_dir.exists():
        return "(empty — knowledge directory does not exist yet)"

    rows: List[str] = []

    def _walk(dir_path: Path, depth: int) -> None:
        try:
            children = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError:
            return
        # Skip _archive (the prompt notes it explicitly to keep noise down).
        children = [
            c for c in children if c.name not in _TREE_EXCLUDE_NAMES and c.name != "_archive"
        ]
        for c in children:
            indent = "  " * depth
            if c.is_dir():
                rows.append(f"{indent}{c.name}/")
                _walk(c, depth + 1)
            elif c.suffix == ".md":
                rows.append(f"{indent}{c.name}")

    _walk(knowledge_dir, 0)

    if not rows:
        return "(empty — no entries yet)"

    if len(rows) > max_lines:
        omitted = len(rows) - max_lines
        rows = rows[:max_lines] + [f"... ({omitted} more lines truncated)"]
    return "\n".join(rows)


def _read_excerpt(path: Path, *, max_chars: int = 400) -> str:
    """Return the first ``max_chars`` of a file, or empty string."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    if len(text) <= max_chars:
        return text.strip()
    return text[:max_chars].strip() + "\n... (truncated)"


def build_library_snapshot(
    knowledge_dir: Path,
    *,
    max_chars: int = LIBRARY_SNAPSHOT_MAX_CHARS,
) -> str:
    """Compose the library snapshot block the Librarian sees.

    Sections (each preceded by a heading):
      - tree listing
      - root README excerpt
      - top-level folder READMEs (one per direct subdirectory of
        ``knowledge_dir``, e.g. ``global/README.md``,
        ``projects/README.md``)
      - index.md excerpt

    The whole block is hard-capped at ``max_chars`` (default ~3 KB).
    Truncation is signalled inline.
    """
    parts: List[str] = []

    # Tree
    parts.append("## Tree\n```\n" + _tree_listing(knowledge_dir) + "\n```")

    # Root README
    root_readme = _read_excerpt(knowledge_dir / "README.md", max_chars=400)
    if root_readme:
        parts.append("## knowledge/README.md (excerpt)\n" + root_readme)

    # Top-level folder READMEs
    if knowledge_dir.exists():
        try:
            subdirs = sorted(
                [
                    p
                    for p in knowledge_dir.iterdir()
                    if p.is_dir() and p.name != "_archive" and not p.name.startswith(".")
                ]
            )
        except OSError:
            subdirs = []
        for sd in subdirs:
            r = _read_excerpt(sd / "README.md", max_chars=300)
            if r:
                parts.append(f"## {sd.name}/README.md (excerpt)\n{r}")

    # index.md
    idx = _read_excerpt(knowledge_dir / "index.md", max_chars=800)
    if idx:
        parts.append("## index.md (excerpt)\n" + idx)
    elif not knowledge_dir.exists() or not (knowledge_dir / "index.md").exists():
        parts.append("## index.md\n(empty — no catalog yet)")

    out = "\n\n".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n\n... (snapshot truncated to fit prompt budget)"
    return out


# ── Knowledge-store derived inputs (kept for the interrupt path) ──────


def read_index_md(knowledge_dir: Path) -> str:
    """Return the contents of ``knowledge/index.md``, or a sentinel."""
    p = knowledge_dir / "index.md"
    try:
        return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "(empty — no entries yet)"
    except OSError as e:
        log.warning("could not read %s: %s", p, e)
        return "(unavailable)"


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        fm: object = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(fm, dict):
        return {}
    fm_typed = cast(Dict[Any, Any], fm)
    return {str(k): v for k, v in fm_typed.items()}


def _split_applies_when(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        raw_list = cast(List[Any], raw)
        return [str(x).strip() for x in raw_list if str(x).strip()]
    if isinstance(raw, str):
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return []


_APPLIES_WHEN_SKIP_TOP = {"index.md", "log.md", "README.md"}


def _is_listable_entry(md: Path, knowledge_dir: Path) -> bool:
    """True iff ``md`` is a candidate entry file for the applies-when table."""
    if "_archive" in md.parts:
        return False
    if md.name == "README.md":
        return False
    if md.parent == knowledge_dir and md.name in _APPLIES_WHEN_SKIP_TOP:
        return False
    return True


def _applies_when_rows_for(md: Path) -> List[str]:
    """Return the ``<id> | <scope> | <phrase>`` rows for one entry. Empty
    list if the file is unreadable, has no frontmatter, isn't confirmed,
    or has no applies-when phrases."""
    try:
        text = md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    fm = _parse_frontmatter(text)
    if not fm:
        return []
    if str(fm.get("status") or "").lower() != "confirmed":
        return []
    lesson_id = str(fm.get("id") or md.stem)
    scope = str(fm.get("scope") or "global")
    phrases = _split_applies_when(fm.get("applies-when") or fm.get("applies_when"))
    return [f"{lesson_id} | {scope} | {phrase}" for phrase in phrases]


def build_applies_when_table(knowledge_dir: Path) -> str:
    """Walk every CONFIRMED entry and emit one line per applies-when
    phrase: ``<lesson_id> | <scope> | <applies-when phrase>``."""
    if not knowledge_dir.exists():
        return "(empty — no confirmed entries)"
    rows: List[str] = []
    for md in sorted(knowledge_dir.rglob("*.md")):
        if not _is_listable_entry(md, knowledge_dir):
            continue
        rows.extend(_applies_when_rows_for(md))
    if not rows:
        return "(empty — no confirmed entries)"
    return "\n".join(rows)


# ── Project slug ──────────────────────────────────────────────────────


def _iter_events_reversed(snapshot: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield events from a snapshot, most recent first.

    Wraps the nested ``turns -> events`` walk so callers don't have to
    re-do the ``isinstance(list)`` / ``isinstance(dict)`` dance. Only
    well-shaped dict events are yielded; everything else is skipped.
    """
    raw_turns_obj: object = snapshot.get("turns") or []
    if not isinstance(raw_turns_obj, list):
        return
    raw_turns = cast(List[Any], raw_turns_obj)
    for turn_obj in reversed(raw_turns):
        turn_map = _as_str_map(turn_obj)
        raw_events_obj: object = turn_map.get("events") or []
        if not isinstance(raw_events_obj, list):
            continue
        raw_events = cast(List[Any], raw_events_obj)
        for ev_obj in reversed(raw_events):
            ev_map = _as_str_map(ev_obj)
            if ev_map:
                yield ev_map


def derive_project_slug(snapshot: Mapping[str, Any]) -> str:
    """Return the *canonical project slug* (git-URL flattened form or
    cwd basename fallback) — what ``current_project_slug()`` on the
    hook side produces. This is the project's identity, not its
    directory name. Use :func:`derive_project_bucket` when you need the
    on-disk bucket directory the librarian + scholar should write to.
    """
    explicit = snapshot.get("project_slug")
    if isinstance(explicit, str) and explicit.strip():
        return _slugify(explicit)
    for ev in _iter_events_reversed(snapshot):
        pl = _as_str_map(ev.get("payload"))
        cand = pl.get("project_slug") or pl.get("slug")
        if isinstance(cand, str) and cand.strip():
            return _slugify(cand)
    cwd = snapshot.get("cwd")
    if isinstance(cwd, str) and cwd:
        return _slugify(os.path.basename(cwd.rstrip("/")) or "unknown")
    return "unknown"


def _snapshot_cwd(snapshot: Mapping[str, Any]) -> Optional[str]:
    """Pull the user's cwd off a buffer snapshot.

    Mirrors the slug-extraction order: top-level field first, then a
    sweep of recent event payloads (most recent wins). Returns ``None``
    when no cwd is recoverable — caller should then skip bucket
    resolution and use the slug verbatim.
    """
    top = snapshot.get("cwd")
    if isinstance(top, str) and top.strip():
        return top
    for ev in _iter_events_reversed(snapshot):
        pl = _as_str_map(ev.get("payload"))
        cand = pl.get("cwd")
        if isinstance(cand, str) and cand.strip():
            return cand
        ev_cwd = ev.get("cwd")
        if isinstance(ev_cwd, str) and ev_cwd.strip():
            return ev_cwd
    return None


def derive_project_bucket(snapshot: Mapping[str, Any]) -> str:
    """Return the on-disk bucket directory name that all components —
    librarian's path proposals, scholar's writes, priming's scope
    boost, the nudge filter — should use for this session's project.

    Calls :func:`aliases.session_bucket` so the bucket name comes from
    the same single-source-of-truth resolver every other call site uses.
    Falls back to the slug verbatim when the snapshot has no cwd or the
    resolver returns ``None``, so older snapshots / mid-session edge
    cases still produce something usable.
    """
    slug = derive_project_slug(snapshot)
    cwd_str = _snapshot_cwd(snapshot)
    if not cwd_str:
        return slug
    try:
        bucket = session_bucket(_home(), Path(cwd_str), slug)
    except Exception:
        bucket = None
    return bucket or slug


_SLUG_NONALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    s2 = _SLUG_NONALNUM.sub("-", s.lower()).strip("-")
    return s2 or "unknown"


# ── Prompt template ───────────────────────────────────────────────────


_PROMPT_TEMPLATE = """\
You are the Librarian role in a two-tier curator system for a personal \
coding-agent memory store.

This memory is NOT just a rulebook. It is a record of **what the user \
prefers, how they think, what they've corrected before, what they've \
asked you to remember, and what they expect of you in this codebase and \
in general**. Preferences are the bulk of it. Hard rules are a subset. \
The user wants this memory to feel like an assistant who remembers them \
— not a compliance system.

You are the active organiser. You read the conversation buffer, you look \
at the existing library, and you PROPOSE a list of structural actions to \
keep the library well-organised and useful. You never write to disk; the \
Scholar (a more capable model) is the gatekeeper and the only writer. \
The Scholar will either APPROVE-and-execute each proposed action or \
VETO-and-drop it.

There is no second chance per pass: vetoed proposals are dropped, the \
Librarian does not retry. But signal that recurs across sessions WILL be \
re-seen. So your job is to be a **generous recall layer** — surface \
anything that looks worth remembering and let the Scholar do the \
precision work.

═══════════════════════════════════════════════════════════════════
THE SALIENCE TEST (cognitive-science framing)
═══════════════════════════════════════════════════════════════════

Memory in humans is gated by **prediction error** — what doesn't match \
your existing model gets encoded; what confirms expectation gets \
compressed. Brains don't store "useful things," they store "things that \
don't match what I already expected." Use the same gate.

For every candidate from the buffer, classify the salience signal:

  **CONTRADICTS an existing entry** (highest priority)
    The user just said something incompatible with what the library already holds.
    Examples:
      - Library has "use celery for background jobs"; user says "actually \
I've switched to rq for this repo"
      - Library has "deploy via staging branch"; user says "we deploy \
direct to prod now for hotfixes"
    Action: propose ``deprecate_entry`` on the older + ``update_entry`` \
(or new ``write_entry``) for the new. Always cite the contradicted entry \
in ``existing_entry``.

  **NOVEL** (capture — would not be derivable from your baseline \
knowledge)
    The candidate is not in the library AND a competent generalist \
assistant would NOT have produced it unprompted.
    Examples:
      - User asserts a stronger version of a default: "use uv for python" \
→ you'd know; "use uv ALWAYS even for ad-hoc scripts, never pip" → \
that's a strict override of the default; novel.
      - User reveals a fact only they know: "the prod DB is in us-west2", \
"I use tmux because my window manager doesn't restore sessions"
      - User-specific preference about agent behaviour: "don't summarise \
the diff at the end", "always ask before deploying"
      - Project-specific convention distinct from how the same problem is \
solved elsewhere: "in this repo, all API clients go through \
PaymentsClientFactory"
    Action: propose ``write_entry``. Set ``salience_signal: "novel"``.

  **REINFORCES an existing entry** (don't write — increment the \
existing's counter)
    The candidate restates something the library already covers, perhaps \
in different wording.
    Examples:
      - Library has "use uv for python"; user says "yeah uv is the way" \
— reinforcement, not new info
    Action: usually propose NOTHING (the daemon will detect the \
reinforcement separately and bump the existing entry's confidence). If \
the new phrasing meaningfully strengthens or extends the entry, propose \
``update_entry`` with ``salience_signal: "reinforces"`` and cite \
``existing_entry``. Otherwise omit.

  **USED-HELPFULLY — the assistant relied on a surfaced entry** (count \
the hit; do NOT rewrite the entry)
    The assistant actually USED an existing library entry to answer THIS \
turn — it agreed with it and applied its content, or explicitly cited it \
("yes, as the memory on X says…", "per our convention, I'll use uv"). \
This is a *positive reliance* signal — the entry earned its keep by \
firing usefully in real work.
    Examples:
      - Library has "use uv for python"; the assistant says "I'll set this \
up with uv as per your convention" and does so → used_helpfully.
      - A surfaced entry's rule shaped the assistant's answer and it acted \
on it.
    NOT used-helpfully:
      - The assistant merely MENTIONED the entry without relying on it.
      - The assistant DISAGREED with or corrected the entry — that is \
``contradicts`` (which routes to fixing the entry), never this. These are \
mutually exclusive: ``used_helpfully`` = the entry was right and helped; \
``contradicts`` = the entry is wrong.
    Action: propose ``update_entry`` (or, if no edit is warranted, the \
lightest action you can) with ``salience_signal: "used_helpfully"``, cite \
the relied-upon entry in ``existing_entry``, AND set ``cited_turn_seq`` to \
the STABLE ``(turn_seq=S)`` value rendered on the buffer line where the \
reliance happened. The daemon bumps that entry's ``fired-helpful`` \
counter once per cited turn — it does NOT rewrite the entry unless your \
proposal also carries substantive new content the Scholar approves. \
Cite the turn precisely: re-seeing the same turn on a later scan must not \
double-count, which only works if you give the stable ``turn_seq``.

  **DRIFT — reconsolidate an entry that was just retrieved** (mutate in \
place, sparingly)
    Retrieval makes a memory labile: an entry that surfaced and was USED \
this turn is the right moment to fold in a genuine refinement — but \
reconsolidation also DISTORTS if you let it (every re-write risks eroding \
the original claim). Propose ``drift`` ONLY when there is real signal, \
never as a reflexive tidy-up.
    Propose when EITHER:
      - the use-context carries a genuine new qualifier, edge case, or \
correction that belongs in the entry (e.g. entry says "use uv"; this turn \
established "…except in the legacy `tools/` dir that pins pip" → integrate \
that caveat), OR
      - the entry can be sharpened without changing its claim (a buried \
rule pulled to the top, a stale tangent cut).
    Hard rules:
      - The load-bearing claim MUST survive intact. You are integrating or \
sharpening, NOT rewriting from scratch. If you can't keep the core claim \
verbatim-in-spirit, this is ``contradicts``, not ``drift``.
      - **RE-TITLE when the body's STATE changes.** If your mutation changes \
what the entry reports about its subject's status — most often flipping an \
open problem to fixed/resolved/implemented, or a doubt to settled — you \
MUST re-derive the ``title:`` frontmatter AND the ``# H1`` heading (and \
re-check ``applies-when``/``keywords``/``summary``) so the index card \
describes the NEW state, then carry them in ``new_body``. A problem-framed \
title sitting on a body that now says the problem is solved is the single \
most common drift defect — the card ends up lying about its own contents. \
This does NOT violate the load-bearing-claim rule above: that rule governs \
the CLAIM, not its framing. Reframing "X is a blind spot" into "X was \
fixed by Y; the lesson is Z" is REQUIRED here, not a forbidden \
rewrite-from-scratch.
      - Pure rephrasing that adds no information and sharpens nothing → \
DON'T. That is exactly the churn that turns memories to mush over time.
      - **SPLIT instead of cramming when the entry sprawls.** Two cases: \
(a) integrating the new nuance would push the entry to cover more than one \
coherent claim, or (b) the entry you are already reconsolidating has \
ALREADY grown large and spans mixed topics. In either case decompose — \
but gently: only when the topics are genuinely separable, never as \
reflexive tidying. You are the curator; reorg is your job. Spin the \
separable material into its OWN entry and trim the original: emit an \
``update_entry`` on the original (leaner body, with a ``[[link]]`` to the \
new entry) PLUS a ``write_entry`` for the spun-off entry. Don't let \
reconsolidation balloon a single file. (If the split pushes a folder over \
its entry cap, the normal reorg pass handles that.) You only learn an \
entry's true size by ``read_entry``-ing it — so when you reconsolidate, \
read the whole entry first and judge whether it has quietly become several \
topics.
    Action: for an in-place refinement, propose ``update_entry`` with \
``salience_signal: "drift"``, cite the entry in ``existing_entry``, and \
put the FULL mutated body (frontmatter + prose) in ``new_body``. For a \
split, emit the trimming ``update_entry`` (also ``salience_signal: \
"drift"``) alongside the new ``write_entry``. Independent of \
``used_helpfully`` (that only counts the hit; ``drift`` actually edits the \
entry) — you may emit both for the same entry in one scan.

  **REDUNDANT / no signal** (skip silently — no proposal)
    Things every competent assistant would already produce or that don't carry information:
      - Code-as-code: "added a for loop", "created a class", "fixed indentation"
      - Conversation filler: "ok", "thanks", "let me try X"
      - Tool output: build logs, test results, file contents
      - Generic facts: "Python uses 4-space indents", "git push uploads commits"
    Action: omit. Don't propose anything for these.

  **IF YOU'RE UNSURE WHICH SIGNAL APPLIES**, propose anyway and leave \
``salience_signal: null``. The Scholar will deliberate. **Recall over \
precision** — you are Sonnet-tier; the Scholar is Opus-tier and applies \
a stricter check. Better to surface a maybe-novel candidate and let the \
Scholar veto than to silently drop a real one.

To classify ``contradicts`` / ``reinforces`` honestly, you MUST actually \
search the library first. Call ``bm25_search`` AND ``embedding_search`` on \
the topic, then ``read_entry`` the top hits and decide. BM25 catches exact \
vocabulary matches; embeddings catch paraphrases. Without these searches \
you can't claim ``novel`` truthfully either — you'd just be guessing.

═══════════════════════════════════════════════════════════════════
IMPORTANT: YOU ARE THE ONLY MEMORY SYSTEM HERE
═══════════════════════════════════════════════════════════════════

Other memory tools (Serena, MCP memory servers, ad-hoc structured notes the
assistant scribbles in the conversation) may be running in the user's
session. Their outputs sometimes appear in the assistant's turns as
frontmatter-shaped blocks like ``--- name: X type: feedback ...``.

**Those are not OUR library.** They are noise from other systems. Do not
defer to them — the user does not see them either. If the USER asserted a
preference and another tool happened to also record it elsewhere, we still
file it in OUR library because we're a different store with different
retrieval. Always re-evaluate based on what the USER said, not what the
assistant wrote afterwards.

═══════════════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES (these are what good proposals look like)
═══════════════════════════════════════════════════════════════════

Example 1 — direct user preference:
  Buffer: ``[7] [user] I always prefer to run lint and type checking on my code``
  Proposal: write_entry, path ``global/conventions/run-lint-and-typecheck.md``,
  body filing a stated user preference. The `reasoning` quotes turn [7] verbatim.

Example 2 — interrogative confirmation:
  Buffer: ``[12] [user] you are not allowed to deploy to prod without my say so right?``
  Proposal: write_entry, path ``global/conventions/never-deploy-without-approval.md``.
  This is a user-stated rule despite being phrased as a question. The
  `reasoning` quotes turn [12] verbatim and notes the confirmatory framing.

Example 3 — workflow revelation:
  Buffer: ``[3] [user] I read the diff myself, just tell me what's left to do``
  Proposal: write_entry, path ``global/preferences/no-summary-of-changes.md``.
  The user has flagged a behavioural preference about response shape.

Example 4 — past incident:
  Buffer: ``[5] [user] we leaked an API key via env.example last year — \
only placeholders from now on``
  Proposal: write_entry, path ``global/security/env-example-placeholders.md``,
  citing the incident and the rule. ``status: provisional``, ``confidence: 0.85``.

Example 5 — nothing worth filing:
  Buffer is all bash tool calls, no user preference asserted.
  Proposal list: ``[]``. This is correct — don't invent.

═══════════════════════════════════════════════════════════════════
GROUND RULES
═══════════════════════════════════════════════════════════════════

1. **Keep the library well-organised at every step.** Never let it grow \
into a huge pile of books. If a folder is approaching 5 entries, propose \
a SplitFolder. If two entries are clearly the same lesson, propose a \
MergeEntries. If a folder needs a README, propose UpdateReadme.

2. **Quote and cite.** Every action's `reasoning` field must reference \
either a verbatim turn quote (with the [turn_id]) or a specific path in \
the library. No vague hand-waving.

3. **User-asserted turns (marked [USER-ASSERTED]) carry user-stated \
preferences/rules.** They came in via `/ultan`. Strongly prefer to file \
them — the user has explicitly named the thing. Do not veto them just \
because the wording is short.

4. **Interrogative confirmations ARE assertions.** "You wouldn't X \
right?" is not a question to debate — it's the user implicitly setting \
an expectation. Treat it as a high-trust candidate, not as conversation.

5. **You have four read-only research tools — use them.** The library \
snapshot in this prompt is a teaser; if you suspect an entry already \
covers a candidate, look. The four tools complement each other:

  - ``read_entry(path="...")`` — read one entry's full contents by its \
path relative to the knowledge root. Use to verify a hit before \
proposing an action against it.
  - ``grep_library(pattern="...", path="...")`` — find by **literal \
substring** match in file contents (optionally scoped to a subdir). Use \
for exact strings or known phrasings.
  - ``bm25_search(query="...", k=5)`` — find by **lexical relevance** \
(BM25 over markdown bodies + paraphrases + keywords). Best for queries \
that share vocabulary with stored entries.
  - ``embedding_search(query="...", k=5)`` — find by **semantic \
similarity** (sentence-transformer cosine). Best when the user's phrasing \
differs from how the entry is written ("we shouldn't ship without review" \
vs an entry titled "require PR approval before deploy"). Catches \
paraphrases BM25 misses.

  For "does anything already cover this concept?", default to firing \
BOTH ``bm25_search`` + ``embedding_search`` for the candidate's core \
claim, then ``read_entry`` the top hits to confirm. Keep your tool use \
focused; you are Sonnet-tier and your budget is tight.

6. **Then ``read_entry`` the candidates** the search returned to verify \
they're actually the same thing. Both BM25 and embedding return false \
positives — never propose UpdateEntry/MergeEntries without reading \
the target first.

7. **NEVER quote secrets or credentials.** Buffer text may contain \
API keys, auth tokens, bearer tokens, OAuth client secrets, \
passwords, private keys (``-----BEGIN ... PRIVATE KEY-----``), \
connection strings with embedded passwords, GitHub PATs (``ghp_*``, \
``github_pat_*``), AWS access keys (``AKIA*``), Anthropic / OpenAI \
keys (``sk-*``), JWTs, session cookies, or similar. **NEVER include \
the literal value** in a proposal's body, applies-when, keywords, \
or `reasoning` quote. If a lesson genuinely needs to reference a \
secret, reference its LOCATION or PURPOSE only ("the deploy token \
lives in 1Password under `prod-deploy`"), never its VALUE. When in \
doubt, omit. Memory is plain markdown on disk and often \
git-tracked — assume the worst.

═══════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════

<active_project>
slug: {{PROJECT_SLUG}}
(use this when picking paths; lessons clearly tied to {{PROJECT_SLUG}} → \
``projects/{{PROJECT_SLUG}}/...``; lessons that apply across repos → \
``global/...``)
</active_project>

<rolling_buffer>
{{ROLLING_BUFFER}}
</rolling_buffer>

Each turn is ``[turn_id] (turn_seq=S) [role] <text>``. ``[turn_id]`` is a \
scan-local label you may quote in ``reasoning``; ``(turn_seq=S)`` is a \
STABLE id you MUST echo in ``cited_turn_seq`` when emitting a \
``used_helpfully`` signal (it is how the daemon avoids double-counting a \
re-seen turn). ``[USER-ASSERTED]`` marks turns that arrived via the \
user's /ultan command — file these by default unless they are \
nonsensical.

<library_snapshot>
{{LIBRARY_SNAPSHOT}}
</library_snapshot>

This is your read-only view of the library's current state. If you need \
to verify an entry's contents before proposing an action against it, use \
``read_entry`` with the relative path of an entry you see in the snapshot \
above. Use ``grep_library`` / ``bm25_search`` / ``embedding_search`` to \
find anything you suspect exists but don't see in the snapshot.

═══════════════════════════════════════════════════════════════════
INTEGRITY-REPAIR TASKS (HIGHEST PRIORITY — fix these first)
═══════════════════════════════════════════════════════════════════

<repair_tasks>
{{REPAIR_TASKS}}
</repair_tasks>

If the block above is not empty, the daemon's deterministic post-write \
pass found library invariants it could NOT fix on its own and is handing \
them to you. These are NOT discretionary — propose an action to repair \
each one. They take priority over salience-driven proposals.

Each task names a ``kind`` (the invariant type), a ``file`` (relative to \
``knowledge/`` — for an over-cap dir this is the directory), a ``target`` \
(the offending value, interpreted per kind), and a ``context`` snippet. \
ALWAYS set ``salience_signal: null`` on a repair proposal — these are \
integrity fixes, not salience judgments, and the Scholar will \
verify-and-execute them rather than judge them for novelty. In \
``reasoning``, quote the task's ``file`` and ``target`` and state which fix \
you chose and why. Dispatch on ``kind``:

──────────────────────────────────────────────────────────────────
kind: broken_wikilink  (``target`` = the wikilink that does not resolve)
──────────────────────────────────────────────────────────────────

  1. **Research the intended target.** The broken target usually got the \
PATH wrong, not the concept. Run ``bm25_search`` AND ``embedding_search`` \
on the target's leaf name and the surrounding context, and \
``grep_library`` for the filename. ``read_entry`` the top hits to confirm \
which existing entry the link was meant to point at.

  2. **Then propose exactly ONE of these EXISTING actions** (no new action \
type — the Scholar executes it as a normal proposal):
     - **Link points at the wrong path but the right entry EXISTS** → \
``update_entry`` on ``file`` whose ``new_body`` is the file's full body \
with the broken ``[[target]]`` rewritten to the correct \
``[[full/path/from/knowledge/root]]`` (no ``.md``; trailing ``/`` for a \
folder link). ``read_entry`` ``file`` first so you reproduce its body \
faithfully and change only the link.
     - **The intended target genuinely does NOT exist yet but SHOULD** \
(the link describes a real lesson worth having) → ``write_entry`` creating \
the missing entry at the path the link points to, with proper frontmatter \
and body. The link then resolves.
     - **The link is bogus / the concept isn't worth an entry** → \
``update_entry`` on ``file`` that removes the broken ``[[target]]`` (drop \
the link or replace it with plain descriptive text), preserving the rest \
of the prose. Explain in ``reasoning`` why no target should exist.

Do the link research with the SAME parallel-search discipline as for \
dedup. A repair proposal that guesses the path without searching will \
likely be vetoed.

──────────────────────────────────────────────────────────────────
kind: overcap_dir  (``file``/``target`` = the over-capacity directory)
──────────────────────────────────────────────────────────────────

The directory holds more than 5 entry .md files and must be rebalanced. \
The ``context`` lists every entry currently in it.

  1. **``read_entry`` the entries** (or their frontmatter \
``title``/``keywords`` from the snapshot) to find the natural thematic \
groupings. Use the existing sub-structure of sibling folders as a guide \
for sensible subfolder names.

  2. **Propose exactly ONE rebalancing action:**
     - **The entries split into 2+ coherent themes** → ``split_folder`` on \
``folder_path`` = the over-cap dir, with ``into`` mapping each new \
subfolder NAME → the list of entry paths (relative to ``knowledge/``) that \
move there. Leave no destination subfolder over 5; entries you don't list \
stay put (so the remainder must also be ≤5). The Scholar's \
``move_entries`` call rewrites all inbound wikilinks atomically.
     - **A few entries clearly belong in an EXISTING sibling folder** → one \
``move_entry`` per such entry (``from_path`` → ``to_path``) until the dir \
is back at or below 5. Prefer this when only one or two entries are \
outliers and the rest are cohesive.

  3. Choose subfolder names that read well as a path \
(``global/python/testing/`` not ``global/python/misc-2/``). Do NOT propose \
a split that just shards the dir into ``part-1``/``part-2`` — the grouping \
must be meaningful.

──────────────────────────────────────────────────────────────────
kind: bad_frontmatter  (``file``/``target`` = the entry with bad frontmatter)
──────────────────────────────────────────────────────────────────

The entry's YAML frontmatter is missing, unparseable, or short the \
required fields (``context`` names the exact defect). The body content is \
fine — only the frontmatter block needs repair.

  1. **``read_entry`` ``file``** to recover its current frontmatter \
(whatever is salvageable) and its full body.

  2. **Propose an ``update_entry``** on ``file`` whose ``new_body`` is the \
entry's UNCHANGED body preceded by a corrected frontmatter block that \
re-serialises valid YAML with EVERY required field present: ``id`` \
(kebab-case matching the filename), ``type``, ``scope`` (consistent with \
the path — ``global`` under ``global/``, ``project:<slug>`` under \
``projects/<slug>/``), ``status``, ``confidence``, ``applies-when``, \
``keywords``, ``title``, ``created``, ``updated``, ``fired``, \
``fired-helpful``, ``sources``. Preserve any existing valid values; fill \
missing fields from the body's content and sensible defaults (e.g. \
``status: provisional``, ``confidence: 0.6``, ``fired: 0``, \
``fired-helpful: 0``). Do NOT rewrite the body prose.

═══════════════════════════════════════════════════════════════════
HIERARCHY INVARIANTS (the Scholar will veto violations)
═══════════════════════════════════════════════════════════════════

  - Every directory must have a README.md after your actions complete.
  - No flat directory may end up with more than 5 entry .md files \
(excluding README). If a WriteEntry would push a folder to 6, propose a \
SplitFolder in the same response.
  - Every wikilink must resolve. Strict format rules:
      • Entry link: full path from knowledge root, no `.md` suffix.
        ✅ `[[global/python/use-uv-not-pip]]`
        ❌ `[[use-uv-not-pip]]` (bare slug — broken everywhere except a README in the same folder)
      • Folder link (resolves to that folder's README.md): full path WITH a trailing slash.
        ✅ `[[global/python/]]`
        ❌ `[[python]]` (no slash means entry link, will be flagged broken)
      • From a README, you MAY use bare names for siblings in the SAME \
folder (`[[use-uv-not-pip]]`, `[[python/]]`). In `index.md` and in any \
other file, ALWAYS use the full path.
  - Every entry's frontmatter must validate (id, type, scope, status, \
confidence, applies-when, keywords, title, created, updated, fired, \
fired-helpful, sources). See the schema in the existing entries.

═══════════════════════════════════════════════════════════════════
ACTION TYPES (these are the only legal `action` values)
═══════════════════════════════════════════════════════════════════

{{ACTION_TYPES}}

Notes that the auto-generated table above doesn't cover:
  - For ``update_readme``: write PROSE only. **Do NOT list child entries \
or sub-folders in the body** — the daemon auto-maintains a \
`<!-- ULTAN:children (auto) -->` block with the live child listing. Your \
prose goes ABOVE that block. Only propose this action when the folder's \
purpose/description needs updating, not just because contents changed.
  - For ``split_folder``: only propose when a folder has >5 entry .md \
files that cluster into clear sub-topics. Don't pre-emptively split a \
folder of 2.
  - For ``deprecate_entry`` (CONFLICT RESOLUTION): when you find two \
entries that contradict each other on the same topic, the user has \
changed their mind. Determine which one is more recent (compare \
``updated:`` frontmatter dates), then propose ``deprecate_entry`` on the \
OLDER with ``superseded_by`` pointing at the newer's path. Also propose \
``update_entry`` on the newer to include a "Supersedes earlier guidance \
at [[old-path]]" sentence in its body so the history is preserved. \
Prefer ``deprecate_entry`` over ``archive_entry`` here — the user may \
want to see what they used to think.
  - For ``abstract_entries`` (REFLECTIVE ABSTRACTION — propose RARELY): \
synthesise a higher-order PARENT rule abstracted from ≥2 related LEAF \
entries. The children STAY in place (still individually retrievable); the \
new parent links to them and the daemon adds a reverse backlink into each. \
This is NOT dedup (``merge_entries`` archives) and NOT reorg \
(``split_folder``/``move_entry`` relocate) — nothing is archived or moved. \
You supply ``child_paths`` (≥2 EXISTING entry paths), ``parent_path`` \
(where the parent lives, ``.md``), ``parent_title``, and ``parent_body`` \
(full markdown with YAML frontmatter using ``type: abstraction`` and a \
``[[wikilink]]`` to each child). Use the bm25/embedding tools to find the \
related leaves; read each before proposing.

    **THE AHA GATE — propose ONLY when ALL FOUR hold (this is a high bar; \
most clusters fail it):**
      1. **Remote children** — the leaves come from DIFFERENT \
domains/contexts (e.g. the python ecosystem vs the js ecosystem), so the \
connection is non-obvious. Same-folder / same-surface groupings almost \
never qualify.
      2. **Predictive lift** — the parent rule lets you make a CONFIDENT \
call on an UNSEEN case that no single child covers.
      3. **Non-obvious** — a competent assistant would NOT have stated this \
rule unprompted (the same surprise bar you use for leaf writes). If you'd \
volunteer it from baseline knowledge, it's not an aha.
      4. **Compresses** — the rule is SHORTER than its children and \
regenerates them.

    GOOD example: ``global/python/likes-lint`` (likes lint in python) + \
``global/js/likes-lint`` (likes lint in js) → parent **"user likes \
linting across languages"** — predicts they'll want lint configured in a \
NEW language like Rust (predictive lift), connects two ecosystems (remote), \
and isn't something you'd assert unprompted (non-obvious).

    MUST-REJECT examples (do NOT propose these — they fail the gate):
      - "these entries are all about yellow things" — same-surface \
grouping, zero predictive lift.
      - "likes uv + likes ruff → likes fast tools" — vague; predicts \
nothing about an unseen case.
      - "likes lint + likes types → likes good code" — true but worthless; \
no actionable prediction.

    When in doubt, DON'T propose. A premature or generic abstraction wastes \
the Scholar's attention and clutters the library. One genuine aha is worth \
more than ten plausible-sounding groupings.

Path conventions:
  - All paths are RELATIVE to ``knowledge/``.
  - Top-level dirs are fixed: ``global/`` (cross-project) and ``projects/<slug>/`` (per-repo).
  - EVERYTHING below those is DYNAMIC — no fixed taxonomy, no preset categories.
    You invent the structure based on what's actually in the library and what's
    being added. Read the <library_snapshot> to see what categories already exist
    and prefer extending them; create new ones when nothing fits.
  - **Every entry lives under a topical subdir, never flat under ``global/`` or
    ``projects/<slug>/``.** A real library has sections; ours does too. Even
    the very first Python lesson goes under ``global/python/`` (or whatever
    sub-topic name fits best), not loose under ``global/``. The point is that
    when the next Python lesson arrives, it has a home; the user can see what
    kinds of things are in the library by looking at folder names.
  - Folder names are dynamic — derive them from the dominant concept of the
    entry (e.g. ``python``, ``api-design``, ``error-handling``, ``testing``,
    ``deployment``, ``conventions``). Lowercase, kebab-case, singular noun
    where natural. Look at what's already in the snapshot and prefer
    extending an existing folder over creating a new one — but if nothing
    fits, create one.
  - When a folder reaches >5 entries that cluster into clearer sub-topics,
    propose a ``split_folder`` action to introduce a deeper level. Don't
    pre-emptively go three deep; let the structure earn its depth as
    content arrives.
  - Every folder must have a README.md. The DAEMON auto-creates missing
    READMEs and auto-maintains the child listing inside a
    ``<!-- ULTAN:children (auto) -->`` block — you do NOT need to propose
    update_readme just to register a new entry or sub-folder. Only propose
    update_readme when the README's PROSE (description of what this
    folder is for) genuinely needs to change.

═══════════════════════════════════════════════════════════════════
WHAT NOT TO PROPOSE
═══════════════════════════════════════════════════════════════════

  - "We renamed foo to bar" → not a durable lesson. Omit.
  - "Let's try option B" → transient decision. Omit.
  - "I think X is nicer than Y" → taste, no rule. Omit.
  - Pure restatements of common knowledge ("Python uses 4-space indents"). Omit.

═══════════════════════════════════════════════════════════════════
ENTRY BODY TEMPLATE (use this when writing or updating)
═══════════════════════════════════════════════════════════════════

```
---
id: <kebab-slug — must equal filename without .md>
type: lesson
scope: global   OR   project:<slug>
status: provisional
confidence: 0.7
applies-when: |
  <situation 1 in which the rule fires>
  <situation 2 ...>
keywords: [<at least 3>]
title: "Human-readable title"
created: YYYY-MM-DD
updated: YYYY-MM-DD
fired: 0
fired-helpful: 0
sources:
  - "user-asserted via /ultan on YYYY-MM-DD"     # or daily/transcript anchor
---

# Title

[2–4 sentence core explanation, encyclopedia voice]

## Details

[More on the rule, when it applies, how to use it.]
```

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════

After any read_entry/grep_library/bm25_search/embedding_search calls you \
need, RETURN a ``LibrarianProposal`` via the structured-output mechanism — \
do NOT print JSON in your message text. The shape (generated from the \
Pydantic model the daemon validates against, so it can't drift) is:

{{RESPONSE_SHAPE}}

If you have nothing to propose, return empty lists for both ``proposals`` \
and ``interrupts``. Empty is a valid output. **Don't make things up just \
to fill the list — but DO surface anything that looks like a real \
preference, paradigm, convention, or workflow pattern. The Scholar is the \
precision filter.**

Boundary rules the daemon enforces on your output (a violation bounces \
back for you to fix, so get them right the first time): every entry path \
is RELATIVE to the knowledge root and ends in ``.md`` (no absolute paths, \
no ``..``); any body you supply on a write/update/merge must have a \
parseable YAML frontmatter block.

═══════════════════════════════════════════════════════════════════
APPLIES-WHEN TABLE (for interrupt candidates only)
═══════════════════════════════════════════════════════════════════

<applies_when_table>
{{APPLIES_WHEN_TABLE}}
</applies_when_table>

Scan the buffer against these phrases. If a buffer turn matches a phrase \
(intent, not literal substring), emit an interrupt candidate. Score 0.5+ \
only. Max 5 interrupts per run.

═══════════════════════════════════════════════════════════════════
END OF PROMPT — do your read_entry/grep_library/bm25_search/\
embedding_search inspection if needed, then RETURN your typed \
LibrarianProposal.
═══════════════════════════════════════════════════════════════════
"""


def load_prompt_template() -> str:
    return _PROMPT_TEMPLATE


_NO_REPAIR_TASKS = "(none — no integrity-repair tasks this run)"


def format_repair_tasks(tasks: Sequence[repair_queue.RepairTask]) -> str:
    """Render drained integrity-repair tasks as the ``<repair_tasks>`` body.

    One numbered block per task, listing kind/file/target/context so the
    Librarian can research and repair each one. Returns a sentinel when
    there are no tasks so the prompt block is never blank."""
    if not tasks:
        return _NO_REPAIR_TASKS
    lines: List[str] = []
    for i, t in enumerate(tasks, start=1):
        lines.append(f"{i}. kind: {t.kind}")
        lines.append(f"   file: {t.file}")
        lines.append(f"   target: {t.target}")
        if t.context:
            lines.append(f"   context: {t.context}")
    return "\n".join(lines)


def assemble_prompt(
    *,
    project_slug: str,
    rolling_buffer: str,
    library_snapshot: str,
    applies_when_table: str,
    repair_tasks: str = _NO_REPAIR_TASKS,
) -> str:
    """Substitute placeholders into the prompt template.

    ACTION_TYPES and RESPONSE_SHAPE are generated from ``_schemas.py``
    at call time so the prompt instructions can never drift from the
    Pydantic models the parser actually validates against. ``repair_tasks``
    is the pre-rendered ``<repair_tasks>`` body (see
    :func:`format_repair_tasks`); it defaults to the empty sentinel so
    callers that don't escalate anything need not pass it.
    """
    out = load_prompt_template()
    for needle, value in (
        ("{{PROJECT_SLUG}}", project_slug or "unknown"),
        ("{{ROLLING_BUFFER}}", rolling_buffer or "(empty)"),
        ("{{LIBRARY_SNAPSHOT}}", library_snapshot or "(empty)"),
        ("{{REPAIR_TASKS}}", repair_tasks or _NO_REPAIR_TASKS),
        ("{{APPLIES_WHEN_TABLE}}", applies_when_table or "(empty)"),
        ("{{ACTION_TYPES}}", describe_action_types_markdown()),
        ("{{RESPONSE_SHAPE}}", describe_librarian_response_shape()),
    ):
        out = out.replace(needle, value)
    return out


# ── Convenience: flatten + format in one shot ─────────────────────────


def buffer_to_prompt_text(
    snapshot: Mapping[str, Any],
    *,
    max_chars: int = ROLLING_BUFFER_MAX_CHARS,
) -> Tuple[str, List[Tuple[int, int, str, str, bool]]]:
    """Flatten a snapshot and return both the formatted block and the
    raw 5-tuples ``(turn_id, turn_seq, role, text, user_asserted)``.

    The flattened turns are capped to the most-recent ones that fit
    within ``max_chars`` (see :func:`cap_buffer_to_recent`) before
    formatting, so the rolling-buffer block — and therefore the
    Librarian prompt — stays bounded. The returned tuple list is the
    SAME capped window, so downstream consumers (seed text, turn count)
    agree with what the model actually saw."""
    flat = cap_buffer_to_recent(flatten_buffer(snapshot), max_chars=max_chars)
    return format_rolling_buffer(flat), flat


def concatenated_buffer_text(
    flat: Iterable[Tuple[int, int, str, str, bool]],
) -> str:
    """Plain text of all turns, newline-joined. Used as seed-regex input."""
    return "\n".join(text for _tid, _seq, _role, text, _u in flat)
