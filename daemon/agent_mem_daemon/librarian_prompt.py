"""Prompt assembly + response parsing for the Librarian.

Pure functions. No I/O happens at import time. The daemon orchestrator
(``librarian.scan``) wires these together with the SDK call wrapper in
``llm.run_librarian_call`` and the audit machinery in ``runs.py``.

The Librarian is the *active organiser* in the new architecture. It
receives:

  1. The conversation buffer (rolling-buffer snapshot).
  2. A snapshot of the library's current state:
       - directory tree
       - root README excerpt
       - index.md content (truncated)
       - top-level folder READMEs
  3. BM25 dedup near-hits for any regex-extracted seed phrases.

…and emits a `LibrarianProposal` JSON object: a list of typed
`ProposedAction` items. The Librarian also has Read+Glob tools so it
can inspect specific entries before proposing actions; the final text
message it produces must be the JSON.

The Scholar is the gatekeeper that approves/vetoes each proposal —
the Librarian never writes to disk.

See ``docs/LIBRARIAN_PROMPT.md`` for the full spec.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from . import _response_parser
from ._schemas import LibrarianProposal


log = logging.getLogger("agent_mem_daemon.librarian_prompt")


# Hard ceiling on the library snapshot block: we don't want a sprawling
# corpus to push the Librarian's prompt past Haiku's sensible context
# size. ~3 KB per the cost-discipline brief.
LIBRARY_SNAPSHOT_MAX_CHARS = 3 * 1024


# ── Buffer flattening (unchanged from the previous design) ────────────


_TEXT_KEYS: Tuple[str, ...] = ("text", "prompt", "response", "content", "message", "summary")


def _payload_role(ev: Dict[str, Any]) -> str:
    """Map an event to a [role] tag for the Librarian's prompt."""
    pl = ev.get("payload") or {}
    if isinstance(pl, dict):
        explicit = pl.get("role")
        if isinstance(explicit, str) and explicit:
            return explicit.lower()
    typ = ev.get("type") or ""
    if typ == "UserPromptSubmit":
        return "user"
    if typ in ("SessionEnd", "SessionStart"):
        return "system"
    return "assistant"


def _payload_text(ev: Dict[str, Any]) -> str:
    """Pull a string body out of an event payload."""
    pl = ev.get("payload") or {}
    if not isinstance(pl, dict):
        return ""
    for k in _TEXT_KEYS:
        v = pl.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    tool = pl.get("tool") or pl.get("name")
    if isinstance(tool, str):
        args = pl.get("arguments") or pl.get("args") or pl.get("input") or {}
        if isinstance(args, dict) and args:
            keys = ",".join(sorted(args.keys())[:4])
            return f"{tool}({keys})"
        return str(tool)
    return ""


def flatten_buffer(snapshot: Dict[str, Any]) -> List[Tuple[int, str, str, bool]]:
    """Materialise a buffer snapshot as a list of
    ``(turn_id, role, text, user_asserted)``.

    The optional 4-tuple form is new: ``/ultan`` events get
    ``user_asserted=True``. The Librarian treats those as user-stated
    rules and is told to file them rather than veto.

    The turn_id is a 1-based monotonic counter across all events in all
    sealed turns. Stop / SessionEnd marker events are skipped — they
    carry no dialogue.
    """
    out: List[Tuple[int, str, str, bool]] = []
    counter = 0
    for turn in snapshot.get("turns", []) or []:
        for ev in turn.get("events", []) or []:
            typ = ev.get("type")
            if typ in ("Stop", "SessionEnd"):
                continue
            text = _payload_text(ev)
            if not text:
                continue
            pl = ev.get("payload") or {}
            user_asserted = bool(
                isinstance(pl, dict) and pl.get("user_asserted")
            )
            counter += 1
            out.append((counter, _payload_role(ev), text, user_asserted))
    return out


def format_rolling_buffer(flat: Sequence[Tuple[int, str, str, bool]]) -> str:
    """Render the (turn_id, role, text, user_asserted) list as the
    ``<rolling_buffer>`` body.

    User-asserted turns get a ``[USER-ASSERTED]`` prefix so the
    Librarian knows the user explicitly named the rule (treat with
    higher priority).
    """
    if not flat:
        return "(empty — no turns with quotable text)"
    lines = []
    for tid, role, text, user_asserted in flat:
        squashed = " ".join(text.split())
        prefix = "[USER-ASSERTED] " if user_asserted else ""
        lines.append(f"[{tid}] [{role}] {prefix}{squashed}")
    return "\n".join(lines)


# ── Seed extraction (BM25 pre-pass — recall-oriented) ─────────────────


_SEED_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:always|never)\s+[a-z][^.!?\n]{6,200}", re.IGNORECASE),
    re.compile(
        r"\b(?:we|you|i)\s+(?:should|must|need to|have to|ought to)\s+[a-z][^.!?\n]{4,200}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthe\s+(?:fix|gotcha|rule|trick|catch|key|trap|issue|problem)\s+(?:is|was)\s+[^.!?\n]{4,200}",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:don'?t|do not)\s+[a-z][^.!?\n]{4,200}", re.IGNORECASE),
    re.compile(
        r"(?:^|[.!?\n]\s+)\s*(?:use|prefer|avoid|stub|wrap|stop)\s+[a-z][^.!?\n]{6,200}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[a-z][a-z0-9 -]{2,40}\s+(?:pattern|principle|convention|approach)\b[^.!?\n]{0,80}",
        re.IGNORECASE,
    ),
)


def extract_seed_phrases(buffer_text: str, *, max_seeds: int = 12) -> List[str]:
    """Regex over the rolling buffer text. Returns deduped seed phrases."""
    raw: List[str] = []
    seen_norm: set[str] = set()
    for pat in _SEED_PATTERNS:
        for m in pat.finditer(buffer_text):
            phrase = m.group(0).strip(" \t\n.,;:-")
            phrase = " ".join(phrase.split())
            if len(phrase) < 8 or len(phrase) > 240:
                continue
            norm = phrase.lower()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            raw.append(phrase)

    deduped: List[str] = []
    for s in sorted(raw, key=len, reverse=True):
        s_low = s.lower()
        if any(s_low in d.lower() for d in deduped):
            continue
        deduped.append(s)

    by_first = sorted(deduped, key=lambda x: raw.index(x) if x in raw else 1_000_000)
    return by_first[:max_seeds]


# ── BM25 hit attachment ───────────────────────────────────────────────


def attach_bm25_hits(
    seeds: Sequence[str],
    bm25_index,
    *,
    knowledge_dir: Optional[Path] = None,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """For each seed, run BM25 and return ``{seed, hits: [...]}``."""
    out: List[Dict[str, Any]] = []
    if bm25_index is None:
        for seed in seeds:
            out.append({"seed": seed, "hits": []})
        return out
    root = knowledge_dir
    if root is None:
        root = getattr(bm25_index, "knowledge_dir", None)
    for seed in seeds:
        try:
            raw_hits = bm25_index.search(seed, k=k)
        except Exception:
            log.exception("bm25 search failed for seed=%r", seed)
            raw_hits = []
        hit_dicts: List[Dict[str, Any]] = []
        for path, score, _snippet in raw_hits or []:
            p = Path(path)
            try:
                rel = str(p.relative_to(root)) if root else str(p)
            except (TypeError, ValueError):
                rel = str(p)
            hit_dicts.append({
                "entry_id": p.stem,
                "score": round(float(score), 3),
                "path": rel,
            })
        out.append({"seed": seed, "hits": hit_dicts})
    return out


def format_bm25_seeds(seeds_with_hits: Sequence[Dict[str, Any]]) -> str:
    """Render the ``<bm25_seeds>`` block."""
    if not seeds_with_hits:
        return "(none — regex extractor found no candidate seeds)"
    blocks: List[str] = []
    for entry in seeds_with_hits:
        seed = entry.get("seed", "")
        hits = entry.get("hits") or []
        lines = [f'seed: "{seed}"']
        if not hits:
            lines.append("  (no hits)")
        else:
            for i, h in enumerate(hits, 1):
                lines.append(
                    f"  hit {i}: entry_id={h.get('entry_id','?')}  "
                    f"score={h.get('score',0.0)}  path={h.get('path','?')}"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


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
            c for c in children
            if c.name not in _TREE_EXCLUDE_NAMES and c.name != "_archive"
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
                [p for p in knowledge_dir.iterdir()
                 if p.is_dir() and p.name != "_archive" and not p.name.startswith(".")]
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
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}
    return fm if isinstance(fm, dict) else {}


def _split_applies_when(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return []


def build_applies_when_table(knowledge_dir: Path) -> str:
    """Walk every CONFIRMED entry and emit one line per applies-when
    phrase: ``<lesson_id> | <scope> | <applies-when phrase>``."""
    rows: List[str] = []
    if not knowledge_dir.exists():
        return "(empty — no confirmed entries)"
    skip_top = {"index.md", "log.md", "README.md"}
    for md in sorted(knowledge_dir.rglob("*.md")):
        if "_archive" in md.parts:
            continue
        if md.name == "README.md":
            continue
        if md.parent == knowledge_dir and md.name in skip_top:
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        fm = _parse_frontmatter(text)
        if not fm:
            continue
        status = str(fm.get("status") or "").lower()
        if status != "confirmed":
            continue
        lesson_id = str(fm.get("id") or md.stem)
        scope = str(fm.get("scope") or "global")
        for phrase in _split_applies_when(
            fm.get("applies-when") or fm.get("applies_when")
        ):
            rows.append(f"{lesson_id} | {scope} | {phrase}")
    if not rows:
        return "(empty — no confirmed entries)"
    return "\n".join(rows)


# ── Project slug ──────────────────────────────────────────────────────


def derive_project_slug(snapshot: Dict[str, Any]) -> str:
    explicit = snapshot.get("project_slug")
    if isinstance(explicit, str) and explicit.strip():
        return _slugify(explicit)
    for turn in reversed(snapshot.get("turns") or []):
        for ev in reversed(turn.get("events") or []):
            pl = ev.get("payload") or {}
            if isinstance(pl, dict):
                cand = pl.get("project_slug") or pl.get("slug")
                if isinstance(cand, str) and cand.strip():
                    return _slugify(cand)
    cwd = snapshot.get("cwd")
    if isinstance(cwd, str) and cwd:
        return _slugify(os.path.basename(cwd.rstrip("/")) or "unknown")
    return "unknown"


_SLUG_NONALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    s2 = _SLUG_NONALNUM.sub("-", s.lower()).strip("-")
    return s2 or "unknown"


# ── Prompt template ───────────────────────────────────────────────────


_PROMPT_TEMPLATE = """You are the Librarian role in a two-tier curator system for a personal coding-agent memory store.

This memory is NOT just a rulebook. It is a record of **what the user prefers, how they think, what they've corrected before, what they've asked you to remember, and what they expect of you in this codebase and in general**. Preferences are the bulk of it. Hard rules are a subset. The user wants this memory to feel like an assistant who remembers them — not a compliance system.

You are the active organiser. You read the conversation buffer, you look at the existing library, and you PROPOSE a list of structural actions to keep the library well-organised and useful. You never write to disk; the Scholar (a more capable model) is the gatekeeper and the only writer. The Scholar will either APPROVE-and-execute each proposed action or VETO-and-drop it.

There is no second chance per pass: vetoed proposals are dropped, the Librarian does not retry. But signal that recurs across sessions WILL be re-seen. So your job is to be a **generous recall layer** — surface anything that looks worth remembering and let the Scholar do the precision work.

═══════════════════════════════════════════════════════════════════
THE CAPTURE TEST (the one heuristic that matters)
═══════════════════════════════════════════════════════════════════

Ask one question of every candidate: **"Is this non-code-specific knowledge a future session would benefit from?"**

  CAPTURE (yes):
    - Paradigms — factory pattern, dependency injection, hexagonal architecture
    - Testing procedures — "use respx for HTTP mocks", "always integration-test the migration"
    - Tooling preferences — "uv not pip", "ripgrep over grep", "use tmux for long-running"
    - Conventions — naming, file layout, branch model, commit shape
    - Workflow patterns — "I always review the diff before merging", "deploy via staging branch first"
    - Behavioural preferences — "don't summarise at the end", "be terse", "ask before deploying"
    - Past-incident lessons — "we leaked an API key via env.example last year"
    - Anti-patterns the user has flagged — "don't mock the DB", "we tried X, it didn't work"
    - Rules in any phrasing — including interrogative confirmations ("you wouldn't X right?")
    - Project-specific conventions distinct from global ones (file under projects/<slug>/)

  SKIP (no):
    - "The assistant added a for loop" / wrote an if-statement / created a class — that's just code, not knowledge
    - "Renamed foo to bar" / "fixed typo on line 42" — transient
    - "Let me try option B" — task state, not durable
    - Generic facts every engineer knows ("Python uses 4-space indents")
    - Conversation filler ("ok", "thanks", "sure")
    - Tool output (the build log, the test results, the file contents)

**Be generous.** The user wants this library to grow into something rich — a real record of how they think, what they prefer, what they expect. Several proposals per pass is normal and good. You are the recall tier; the Scholar is the precision tier. A vetoed proposal costs nothing; a missed preference costs the user trust in the system.

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
  Buffer: ``[5] [user] we leaked an API key via env.example last year — only placeholders from now on``
  Proposal: write_entry, path ``global/security/env-example-placeholders.md``,
  citing the incident and the rule. ``status: provisional``, ``confidence: 0.85``.

Example 5 — nothing worth filing:
  Buffer is all bash tool calls, no user preference asserted.
  Proposal list: ``[]``. This is correct — don't invent.

═══════════════════════════════════════════════════════════════════
GROUND RULES
═══════════════════════════════════════════════════════════════════

1. **Keep the library well-organised at every step.** Never let it grow into a huge pile of books. If a folder is approaching 5 entries, propose a SplitFolder. If two entries are clearly the same lesson, propose a MergeEntries. If a folder needs a README, propose UpdateReadme.

2. **Quote and cite.** Every action's `reasoning` field must reference either a verbatim turn quote (with the [turn_id]) or a specific path in the library. No vague hand-waving.

3. **User-asserted turns (marked [USER-ASSERTED]) carry user-stated preferences/rules.** They came in via `/ultan`. Strongly prefer to file them — the user has explicitly named the thing. Do not veto them just because the wording is short.

4. **Interrogative confirmations ARE assertions.** "You wouldn't X right?" is not a question to debate — it's the user implicitly setting an expectation. Treat it as a high-trust candidate, not as conversation.

5. **You have three search tools — use them.** The library snapshot in this prompt is a teaser; if you suspect an entry already covers a candidate, look. The three tools complement each other:

  - ``Glob("**/*.md")`` — find by **filename pattern**. Use when you're hunting for a specific path or a folder's contents.
  - ``Grep(pattern="...", path="...")`` — find by **literal regex** match in file contents. Use for exact strings or known phrasings.
  - ``mcp__agent_mem_library__bm25_search(query="...", k=5)`` — find by **content relevance** (BM25 ranking). Use when you have a concept or phrase and want the top-K semantically related entries. This is the right tool for "does anything in the library already cover this idea?" — much better than guessing keywords for Grep.

  Use BOTH bm25 (for relevance) AND glob/grep (for verification) when checking for duplicates: bm25 surfaces candidates, then Read the top hit to confirm. Aim for ~5 tool calls per run; you are Haiku-tier and your budget is tight.

6. **Then Read the candidates** the search returned to verify they're actually the same thing. BM25 false-positives happen — never propose UpdateEntry/MergeEntries without Reading the target first.

═══════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════

<active_project>
slug: {{PROJECT_SLUG}}
(use this when picking paths; lessons clearly tied to {{PROJECT_SLUG}} → ``projects/{{PROJECT_SLUG}}/...``; lessons that apply across repos → ``global/...``)
</active_project>

<rolling_buffer>
{{ROLLING_BUFFER}}
</rolling_buffer>

Each turn is ``[turn_id] [role] <text>``. ``[USER-ASSERTED]`` marks turns that arrived via the user's /ultan command — file these by default unless they are nonsensical.

<library_snapshot>
{{LIBRARY_SNAPSHOT}}
</library_snapshot>

This is your read-only view of the library's current state. If you need to verify an entry's contents before proposing an action against it, use the Read tool with the relative path of an entry you see in the snapshot above. Use Glob (e.g. `Glob("**/*.md")`) to find anything you suspect exists but don't see in the snapshot.

═══════════════════════════════════════════════════════════════════
HIERARCHY INVARIANTS (the Scholar will veto violations)
═══════════════════════════════════════════════════════════════════

  - Every directory must have a README.md after your actions complete.
  - No flat directory may end up with more than 5 entry .md files (excluding README). If a WriteEntry would push a folder to 6, propose a SplitFolder in the same response.
  - Every wikilink must resolve. Strict format rules:
      • Entry link: full path from knowledge root, no `.md` suffix.
        ✅ `[[global/python/use-uv-not-pip]]`
        ❌ `[[use-uv-not-pip]]` (bare slug — broken everywhere except a README in the same folder)
      • Folder link (resolves to that folder's README.md): full path WITH a trailing slash.
        ✅ `[[global/python/]]`
        ❌ `[[python]]` (no slash means entry link, will be flagged broken)
      • From a README, you MAY use bare names for siblings in the SAME folder (`[[use-uv-not-pip]]`, `[[python/]]`). In `index.md` and in any other file, ALWAYS use the full path.
  - Every entry's frontmatter must validate (id, type, scope, status, confidence, applies-when, keywords, title, created, updated, fired, fired-helpful, sources). See the schema in the existing entries.

═══════════════════════════════════════════════════════════════════
ACTION TYPES (these are the only legal `action` values)
═══════════════════════════════════════════════════════════════════

{{ACTION_TYPES}}

Notes that the auto-generated table above doesn't cover:
  - For ``update_readme``: write PROSE only. **Do NOT list child entries or sub-folders in the body** — the daemon auto-maintains a `<!-- ULTAN:children (auto) -->` block with the live child listing. Your prose goes ABOVE that block. Only propose this action when the folder's purpose/description needs updating, not just because contents changed.
  - For ``split_folder``: only propose when a folder has >5 entry .md files that cluster into clear sub-topics. Don't pre-emptively split a folder of 2.
  - For ``deprecate_entry`` (CONFLICT RESOLUTION): when you find two entries that contradict each other on the same topic, the user has changed their mind. Determine which one is more recent (compare ``updated:`` frontmatter dates), then propose ``deprecate_entry`` on the OLDER with ``superseded_by`` pointing at the newer's path. Also propose ``update_entry`` on the newer to include a "Supersedes earlier guidance at [[old-path]]" sentence in its body so the history is preserved. Prefer ``deprecate_entry`` over ``archive_entry`` here — the user may want to see what they used to think.

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

After any Read/Glob/bm25_search calls you need, your FINAL message must be a single JSON object — nothing else, no fences, no commentary, no markdown around it. The schema is:

{{RESPONSE_SHAPE}}

If you have nothing to propose, emit ``{"proposals": [], "interrupts": []}``. Empty is a valid output. **Don't make things up just to fill the list — but DO surface anything that looks like a real preference, paradigm, convention, or workflow pattern. The Scholar is the precision filter.**

═══════════════════════════════════════════════════════════════════
APPLIES-WHEN TABLE (for interrupt candidates only)
═══════════════════════════════════════════════════════════════════

<applies_when_table>
{{APPLIES_WHEN_TABLE}}
</applies_when_table>

Scan the buffer against these phrases. If a buffer turn matches a phrase (intent, not literal substring), emit an interrupt candidate. Score 0.5+ only. Max 5 interrupts per run.

═══════════════════════════════════════════════════════════════════
END OF PROMPT — do your Read/Glob inspection if needed, then emit JSON only, starting with `{` on the final line.
═══════════════════════════════════════════════════════════════════
"""


def load_prompt_template() -> str:
    return _PROMPT_TEMPLATE


def assemble_prompt(
    *,
    project_slug: str,
    rolling_buffer: str,
    library_snapshot: str,
    applies_when_table: str,
) -> str:
    """Substitute placeholders into the prompt template.

    ACTION_TYPES and RESPONSE_SHAPE are generated from ``_schemas.py``
    at call time so the prompt instructions can never drift from the
    Pydantic models the parser actually validates against.
    """
    from ._schemas import (
        describe_action_types_markdown,
        describe_librarian_response_shape,
    )

    out = load_prompt_template()
    for needle, value in (
        ("{{PROJECT_SLUG}}", project_slug or "unknown"),
        ("{{ROLLING_BUFFER}}", rolling_buffer or "(empty)"),
        ("{{LIBRARY_SNAPSHOT}}", library_snapshot or "(empty)"),
        ("{{APPLIES_WHEN_TABLE}}", applies_when_table or "(empty)"),
        ("{{ACTION_TYPES}}", describe_action_types_markdown()),
        ("{{RESPONSE_SHAPE}}", describe_librarian_response_shape()),
    ):
        out = out.replace(needle, value)
    return out


# ── JSON response parsing ─────────────────────────────────────────────


def parse_librarian_json(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse the Librarian's JSON output into a ``LibrarianProposal`` dict.

    Returns the validated response as a plain dict (the
    ``LibrarianProposal.model_dump()`` shape), or ``None`` on any
    failure. The orchestrator turns ``None`` into an empty packet.
    """
    parsed, diag = _response_parser.parse_response(response_text, LibrarianProposal)
    if parsed is None:
        if diag.error:
            log.debug("librarian JSON parse failed: %s", diag.error)
        return None
    if diag.repair_applied:
        log.info("librarian JSON required json-repair to parse")
    return parsed.model_dump()


def normalise_packet(parsed: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Take a parsed Librarian dict and return the two lists the
    scheduler-facing EvidencePacket needs: ``proposals`` and
    ``interrupts``.

    Tolerates the old shape (``candidates``/``interrupt_candidates``)
    by returning empty proposals if those keys are present but
    ``proposals`` is missing — this prevents a half-migrated daemon
    from crashing on rolled-back files.
    """
    proposals_raw = parsed.get("proposals")
    ints_raw = parsed.get("interrupts")
    if ints_raw is None:
        ints_raw = parsed.get("interrupt_candidates")

    def _clean(items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        return [it for it in items if isinstance(it, dict)]

    return {
        "proposals": _clean(proposals_raw),
        "interrupts": _clean(ints_raw),
    }


# ── Convenience: flatten + format in one shot ─────────────────────────


def buffer_to_prompt_text(
    snapshot: Dict[str, Any],
) -> Tuple[str, List[Tuple[int, str, str, bool]]]:
    """Flatten a snapshot and return both the formatted block and the
    raw 4-tuples (turn_id, role, text, user_asserted)."""
    flat = flatten_buffer(snapshot)
    return format_rolling_buffer(flat), flat


def concatenated_buffer_text(
    flat: Iterable[Tuple[int, str, str, bool]],
) -> str:
    """Plain text of all turns, newline-joined. Used as seed-regex input."""
    return "\n".join(text for _tid, _role, text, _u in flat)
