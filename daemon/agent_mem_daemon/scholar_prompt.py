"""Scholar prompt assembly + response parsing.

The Scholar is the gatekeeper + executor. It receives the Librarian's
ProposedAction list and either APPROVES (executes via tool calls) or
VETOES (drops with one-sentence reason). No fix-ups — drop-only
behaviour is user-mandated.

This module owns:
  - Prompt assembly.
  - Final-message JSON parsing.
  - Pure-Python hierarchy invariant checker (deterministic post-write
    validation pass).
  - Nudge file appending (the interrupt path is unchanged).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from . import _response_parser
from ._schemas import ScholarReview
from .librarian import EvidencePacket
from .paths import knowledge_dir, pending_nudges_path


log = logging.getLogger("agent_mem_daemon.scholar_prompt")


# ── Prompt assembly ────────────────────────────────────────────────────


_PROMPT_TEMPLATE = """\
You are the Scholar role in a two-tier curator system for a personal coding-agent memory store.

You are the gatekeeper AND the executor. The Librarian (a cheaper model) has assembled a list of ProposedActions below — restructuring moves it wants applied to the library. For each proposal, you must:

  1. **Verify the claim**: Read the referenced files via your Read/Glob/Grep tools. Trust nothing the Librarian asserts — the snapshot it saw was truncated and it sometimes hallucinates content.
  2. **Decide**: APPROVE (execute the action via Write/Edit) or VETO (drop with a one-sentence reason).
  3. **Execute on approve**: use Write/Edit to actually perform the action.
  4. **Maintain the library state**: update knowledge/index.md and append a line to knowledge/log.md for every approved action.

═══════════════════════════════════════════════════════════════════
HARD RULE: APPROVE-AND-EXECUTE OR VETO-AND-DROP. NO FIX-UPS.
═══════════════════════════════════════════════════════════════════

You may NOT silently rewrite a proposal. If the Librarian's path is wrong, you VETO. If the Librarian's body is too long, you VETO. If two proposals contradict each other, you VETO whichever one is weaker. The Librarian is forced to be careful by this rule — if you start fixing its mistakes, it will get sloppier.

**SAME-PATH DEDUPE (critical — parallel Librarians can collide):** Multiple Librarian workers run concurrently on different sessions, and two of them may independently propose an action against the same target path within a single batch you review. If two or more proposals target the same path, APPROVE the FIRST one (by packet order, then by `action_index`) and VETO every subsequent one with reason "duplicate-target — earlier proposal in same batch already targets <path>". Target paths to check by action type:

  - `write_entry` / `update_entry` / `archive_entry`: ``path``
  - `merge_entries`: ``target_path``
  - `move_entry`: ``to_path``
  - `update_readme`: ``folder_path``
  - `add_wikilink`: ``from_path`` (the file being edited)
  - `split_folder`: ``folder_path``

The lost candidate is not lost forever — if it's a real lesson it will recur in a future session and you'll see fresh evidence. Approving both creates a lost-update race: the second Write tool call overwrites the first.

When you APPROVE, execute the action exactly as proposed (path, body, etc.). The only modifications you may make are:
  - Stamp ``created`` / ``updated`` ISO dates if the Librarian left them as placeholders.
  - Normalise the ``id`` field to match the filename (kebab-case, no .md).
  - Fix obvious YAML frontmatter syntax bugs that would block parsing.

Anything more substantial is a VETO. The lesson will recur in a future session.

═══════════════════════════════════════════════════════════════════
HIERARCHY INVARIANTS — VETO any action that would violate these
═══════════════════════════════════════════════════════════════════

  1. Every directory must have a README.md eventually. **The daemon's
     post-action reconciler creates missing READMEs automatically** —
     do NOT veto a write_entry just because the parent folder has no
     README yet; the reconciler will add one with an auto-managed
     child listing. Only veto on README-related issues when the
     Librarian's proposed UpdateReadme contents are clearly wrong
     (e.g. wrong folder, wrong prose).
  2. No flat directory may end up with more than 5 entry .md files (excluding README). If a write/move would push a folder to 6 and no SplitFolder accompanies it, VETO the write.
  3. Every wikilink must resolve. Wikilinks follow STRICT format rules:
       - To an ENTRY: use the full path from knowledge root, no `.md` suffix.
         ✅ ``[[global/python/use-uv-not-pip]]``
         ❌ ``[[use-uv-not-pip]]``  (bare slug doesn't resolve)
         ❌ ``[[use-uv-not-pip.md]]`` (no .md suffix)
       - To a FOLDER (which points at its README.md): use the full path
         from knowledge root WITH a trailing slash.
         ✅ ``[[global/python/]]``
         ❌ ``[[python]]``  (bare name = entry link, will be broken)
         ❌ ``[[global/python]]``  (no slash = entry link, will be broken)
       - From a README to a sibling entry/folder in the SAME directory you
         MAY use a bare name (``[[use-uv-not-pip]]`` or ``[[python/]]``)
         and the validator will resolve it relative to that README's dir.
         For everything else (especially links in ``index.md``), use the
         full path from knowledge root.
  4. Every entry's frontmatter must validate (required fields: id, type, scope, status, confidence, applies-when, keywords, title, created, updated, fired, fired-helpful, sources).
  5. Paths must agree with scope: ``scope: global`` ⇒ under ``global/...``; ``scope: project:<slug>`` ⇒ under ``projects/<slug>/...``.

═══════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════

<batch_timestamp>
{{ISO_TIMESTAMP}}
</batch_timestamp>

<library_snapshot>
{{LIBRARY_SNAPSHOT}}
</library_snapshot>

<librarian_proposals>
{{PACKETS_JSON}}
</librarian_proposals>

The proposals are a JSON array of Librarian packets accumulated since your last run. Each packet contains:
  - ``session_id`` — which conversation produced it
  - ``proposals`` — list of ProposedAction items in order
  - ``interrupts`` — interrupt nudge candidates (orthogonal — see B below)

Each ProposedAction has an ``action`` discriminator and the corresponding fields. Possible action types (generated from the same Pydantic schema the parser validates against):

{{ACTION_TYPES}}

Special notes:
  - ``update_readme``: **README PROSE only — do NOT include a child listing in the body.** The daemon auto-maintains a `<!-- ULTAN:children (auto) -->` block at the bottom of every README; your prose goes above it. Listing children manually creates duplicate bullet lists.
  - ``deprecate_entry`` execution recipe (the Librarian uses this for conflict resolution — the user changed their mind and a newer entry supersedes the older):
      1. ``Read`` the file at ``path``.
      2. ``Edit`` the frontmatter: set ``status: deprecated`` and add a ``superseded_by: <superseded_by>`` line.
      3. ``Edit`` the body to insert, immediately after the first ``# Heading`` line:
            ``> **Superseded by [[<superseded_by>]] as of <today's ISO date>**. Kept for historical reference.``
      4. Do NOT delete, archive, or otherwise move the file. The wikilinks pointing at it must keep resolving.

All paths are relative to ``knowledge/`` and resolved from your CURRENT WORKING DIRECTORY. Never construct absolute paths — never type ``~/.agent-mem/...`` or ``/Users/.../.agent-mem/...``. Use only relative paths (e.g. ``knowledge/index.md``, ``knowledge/global/python/use-uv.md``). Your cwd is set correctly by the daemon; absolute paths can land you in the wrong store entirely (this has happened in testing).

═══════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════

A. PROPOSALS.

   For each packet in <librarian_proposals>, walk its ``proposals`` list in order. For each proposed action:

     1. Read the referenced files (paths in the action's fields). Verify the Librarian's reasoning matches reality.
     2. Decide APPROVE or VETO. Approve when ALL of:
          - The reasoning cites either a buffer quote or a library path you can verify.
          - The action would not violate a hierarchy invariant.
          - The action genuinely improves the library (vs leaving it alone).
        Veto otherwise.
     3. On APPROVE:
          - Execute the action via Write/Edit/Glob.
          - For ``write_entry`` / ``update_entry`` / ``merge_entries``: write the file at the specified path.
          - For ``move_entry``: read the source, write to the destination, then delete the source (use Write with empty content then verify, or use Bash if available — but Bash is not in your allowed tools, so prefer Read+Write+Edit). When you cannot delete a file with your tools, leave the source as a stub redirect note instead.
          - For ``archive_entry``: read the source, write a copy under ``_archive/<original-relative-path>``, then mark the source's frontmatter ``status: stale`` via Edit (do not delete — archive means moved out of active rotation).
          - For ``update_readme``: write/overwrite the README.md at the folder_path. Write ONLY the prose (heading + description of what this folder is for). Do not write a child-listing — the daemon's reconciler appends `<!-- ULTAN:children (auto) -->` block automatically after every batch.
          - For ``add_wikilink``: use Edit on ``from_path`` to add the link in a Related section.
          - For ``split_folder``: create the new subfolders (with READMEs) and move the listed entries.
          - After every approved action, update knowledge/index.md and append to knowledge/log.md.
     4. On VETO:
          - Do nothing on disk.
          - Append a veto line to knowledge/log.md.

   The order of execution matters for some action types (a SplitFolder followed by a WriteEntry that depends on the new subfolder, for example). Process within a packet in the order given. Across packets, process oldest packet first.

B. INTERRUPTS.

   For each packet's ``interrupts`` list, decide APPROVE or VETO per the rules:
     - APPROVE only if the user is actively in the situation the lesson addresses.
     - VETO if the lesson is provisional, if the user is just reading code, or if the nudge would not be actionable.
     - On APPROVE, phrase the nudge in present tense addressed to the agent (NOT the user). The daemon appends it to ~/.agent-mem/pending-nudges.md server-side; you only declare it in the final JSON.

C. LOG FORMAT.

   Each Scholar action gets one block in knowledge/log.md:

     ## [<ISO-8601 timestamp>] <action> | <status-or-target>
     - Source: <session_id or daily file or transcript anchor>
     - Trigger: Librarian proposal index=<n> session=<session_id>
     - <one-line action-specific note>

   Action values: write | update | merge | move | archive | update-readme | add-wikilink | split-folder | veto | nudge | interrupt-veto

D. FINAL OUTPUT.

   After all tool calls are complete, your LAST message must be a single JSON object — nothing else, no fences, no commentary, no markdown around it. The schema (generated from the same Pydantic models the parser validates against):

{{RESPONSE_SHAPE}}

   ``action_index`` is a flat, 0-based index across ALL packets concatenated in order. Packet 0's proposals[0] is index 0; packet 0's proposals[1] is index 1; if packet 0 had 2 proposals, packet 1's proposals[0] is index 2; and so on. Every proposed action must appear in ``decisions`` exactly once.

   The daemon parses this for queue accounting AND for nudge-file appending. If ``action`` is ``approve`` on an interrupt, supply the user-facing ``text``.

═══════════════════════════════════════════════════════════════════
HEURISTICS
═══════════════════════════════════════════════════════════════════

  - VETO is the default. Approve when you can affirmatively justify it.
  - VETO when the Librarian's body or reasoning contains placeholder text like "TODO" or "<...>".
  - VETO when a proposed write would create a near-duplicate of an existing entry (read both before deciding).
  - VETO when the action's path doesn't match the implied scope (e.g. ``write_entry`` body says ``scope: global`` but path is ``projects/foo/...``).
  - User-asserted content (marked in buffer turns as [USER-ASSERTED]) carries higher trust — lean toward approve when the Librarian's proposal flows directly from such a turn.

═══════════════════════════════════════════════════════════════════
END OF PROMPT — begin by reading knowledge/index.md (if it exists), then walk the proposals.
═══════════════════════════════════════════════════════════════════
"""


def _packets_to_indexed_json(packets: Sequence[EvidencePacket]) -> Tuple[str, int]:
    """Render packets to JSON with a flat ``_action_index`` annotation on
    each proposal, so the Scholar can easily reference indices.

    Returns (json_string, total_proposal_count).
    """
    cursor = 0
    annotated: List[Dict[str, Any]] = []
    for p in packets:
        proposals = p.get("proposals") or []
        annotated_proposals = []
        for prop in proposals:
            if isinstance(prop, dict):
                annotated_proposals.append({"_action_index": cursor, **prop})
                cursor += 1
            else:
                annotated_proposals.append({"_action_index": cursor, "raw": str(prop)})
                cursor += 1
        # Drop legacy keys the scheduler may setdefault for backward
        # compat (``candidates`` is a no-op artifact of the old shape).
        annotated.append({
            **{
                k: v for k, v in p.items()
                if k not in ("proposals", "candidates")
            },
            "proposals": annotated_proposals,
        })
    return json.dumps(annotated, indent=2, ensure_ascii=False), cursor


def build_prompt(
    packets: Sequence[EvidencePacket],
    *,
    now: Optional[datetime] = None,
    library_snapshot: Optional[str] = None,
) -> str:
    """Render the Scholar prompt for a given batch.

    Args:
        packets: list of Librarian EvidencePackets.
        now: override clock for tests; defaults to ``datetime.now(UTC)``.
        library_snapshot: pre-built snapshot string. When None, the
            Scholar builds its own via Read/Glob during the call. We
            still pass an excerpt so it doesn't have to spend a turn
            on it.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    packets_json, _total = _packets_to_indexed_json(packets)
    if library_snapshot is None:
        # Defer to librarian_prompt's snapshot generator — same logic,
        # same truncation policy.
        from . import librarian_prompt as lp
        library_snapshot = lp.build_library_snapshot(knowledge_dir())
    # We use .replace() rather than .format() so the schema-derived
    # blocks (which contain literal `{` and `}` from JSON examples)
    # can be inlined without escaping. The ACTION_TYPES and
    # RESPONSE_SHAPE placeholders are generated from _schemas.py at
    # call time so the prompt instructions can never drift from the
    # Pydantic models the parser validates against.
    from ._schemas import (
        describe_action_types_markdown,
        describe_scholar_response_shape,
    )

    out = _PROMPT_TEMPLATE
    for needle, value in (
        ("{{ISO_TIMESTAMP}}", now.isoformat(timespec="seconds")),
        ("{{LIBRARY_SNAPSHOT}}", library_snapshot),
        ("{{PACKETS_JSON}}", packets_json),
        ("{{ACTION_TYPES}}", describe_action_types_markdown()),
        ("{{RESPONSE_SHAPE}}", describe_scholar_response_shape()),
    ):
        out = out.replace(needle, value)
    return out


# ── Response parsing ───────────────────────────────────────────────────


def parse_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Extract the final-message JSON from the Scholar's response.

    Returns ``(parsed_dict, parsed_ok)``. The dict shape mirrors
    ``ScholarReview.model_dump()`` — ``decisions`` (list) plus
    ``interrupts_processed`` (list).
    """
    parsed, diag = _response_parser.parse_response(response_text, ScholarReview)
    if parsed is None:
        if diag.error:
            log.debug("scholar JSON parse failed: %s", diag.error)
        return None, False
    if diag.repair_applied:
        log.info("scholar JSON required json-repair to parse")
    return parsed.model_dump(), True


def summarise_decisions(parsed: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Roll the Scholar's JSON into a counters dict for the audit row.

    Returns keys like ``approve``, ``veto`` (for proposals) and
    ``nudge``, ``interrupt-veto`` (for interrupts).
    """
    counters: Dict[str, int] = {}
    if not isinstance(parsed, dict):
        return counters
    for item in parsed.get("decisions", []) or []:
        if isinstance(item, dict):
            d = str(item.get("decision", "")).strip().lower()
            if d:
                counters[d] = counters.get(d, 0) + 1
    for item in parsed.get("interrupts_processed", []) or []:
        if isinstance(item, dict):
            action = str(item.get("action", "")).strip().lower()
            if action == "approve":
                counters["nudge"] = counters.get("nudge", 0) + 1
            elif action == "veto":
                counters["interrupt-veto"] = counters.get("interrupt-veto", 0) + 1
            elif action:
                counters[action] = counters.get(action, 0) + 1
    return counters


# ── Nudge file writing (interrupt path — unchanged) ────────────────────


_NUDGE_ID_LEN = 8


def _make_nudge_id() -> str:
    return uuid.uuid4().hex[:_NUDGE_ID_LEN]


def render_nudge_block(
    *,
    nudge_id: str,
    created: str,
    lesson_path: str,
    text: str,
) -> str:
    safe_text = text.strip() or "(no text supplied)"
    return (
        "---\n"
        f"id: {nudge_id}\n"
        f"created: {created}\n"
        f"lesson: {lesson_path}\n"
        "---\n"
        f"{safe_text}\n"
    )


def append_nudges_from_response(
    parsed: Optional[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Append approved interrupts to the pending-nudges file.

    Returns the list of nudge dicts actually written. Same semantics as
    the previous design — kept stable so the hook layer doesn't need
    changes.
    """
    if not isinstance(parsed, dict):
        return []
    interrupts = parsed.get("interrupts_processed") or []
    if not isinstance(interrupts, list):
        return []
    if now is None:
        now = datetime.now(timezone.utc)
    target = path if path is not None else pending_nudges_path()

    written: List[Dict[str, str]] = []
    blocks: List[str] = []
    for item in interrupts:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().lower()
        if action != "approve":
            continue
        text = str(item.get("text") or "").strip()
        lesson_path = str(item.get("lesson_path") or item.get("lesson_id") or "").strip()
        if not text or not lesson_path:
            log.warning(
                "skipping malformed nudge approval (missing text or lesson): %r",
                item,
            )
            continue
        nudge_id = _make_nudge_id()
        block = render_nudge_block(
            nudge_id=nudge_id,
            created=now.isoformat(timespec="seconds"),
            lesson_path=lesson_path,
            text=text,
        )
        blocks.append(block)
        written.append({
            "id": nudge_id,
            "lesson_path": lesson_path,
            "text": text,
        })

    if not blocks:
        return []

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(target), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, "".join(blocks).encode("utf-8"))
        finally:
            os.close(fd)
    except OSError as e:
        log.warning("could not append nudges to %s: %s", target, e)
        return []
    return written


def parse_nudges_file(text: str) -> List[Dict[str, str]]:
    """Parse a pending-nudges.md body into a list of nudge dicts."""
    if not text or not text.strip():
        return []
    out: List[Dict[str, str]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        while i < n and lines[i].strip() != "---":
            i += 1
        if i >= n:
            break
        i += 1
        meta: Dict[str, str] = {}
        while i < n and lines[i].strip() != "---":
            line = lines[i]
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
            i += 1
        if i >= n:
            break
        i += 1
        body_lines: List[str] = []
        while i < n and lines[i].strip() != "---":
            body_lines.append(lines[i])
            i += 1
        body = "\n".join(body_lines).strip()
        if meta or body:
            out.append({
                "id": meta.get("id", ""),
                "created": meta.get("created", ""),
                "lesson": meta.get("lesson", ""),
                "text": body,
            })
    return out


# ── Hierarchy invariants (deterministic post-write validation) ────────


# Frontmatter fields the schema requires on every entry.
_REQUIRED_FRONTMATTER_FIELDS = (
    "id", "type", "scope", "status", "confidence",
    "applies-when", "keywords", "title",
    "created", "updated", "fired", "fired-helpful", "sources",
)


# Wikilink regex — matches [[path/to/entry]] and [[path/to/entry|alias]].
_WIKILINK_RE = re.compile(r"\[\[([^\]]+?)\]\]")


# How many entry .md files per directory we permit before flagging.
MAX_FLAT_DIR_ENTRIES = 5


def check_invariants(knowledge_dir_path: Path) -> List[str]:
    """Walk the knowledge tree and return a list of invariant violations.

    Each violation is a one-line human-readable string. Empty list means
    everything is well-formed. The Scholar calls this after executing
    approved actions and logs WARN on each violation; this is the safety
    net for "the Scholar should have caught it" cases.

    Checks:
      1. Every directory has a README.md.
      2. No directory has >MAX_FLAT_DIR_ENTRIES entry .md files.
      3. Every wikilink resolves.
      4. Every entry's frontmatter has the required fields.
    """
    out: List[str] = []
    if not knowledge_dir_path.exists():
        return out

    all_paths: set[Path] = set()
    entry_files: List[Path] = []
    all_md_files: List[Path] = []   # includes READMEs, index.md, log.md

    # Collect all .md files. Wikilink + body checks scan all of them;
    # frontmatter / per-dir-count checks only apply to entries.
    for md in sorted(knowledge_dir_path.rglob("*.md")):
        if "_archive" in md.parts:
            continue
        all_paths.add(md.resolve())
        all_md_files.append(md)
        if md.name in ("README.md", "index.md", "log.md"):
            continue
        entry_files.append(md)

    # Check 1: every directory under knowledge_dir has a README.md.
    # Walk all dirs that contain at least one entry .md OR a sub-README.
    dirs_to_check: set[Path] = set()
    for md in entry_files:
        d = md.parent
        # Walk up to (and including) the knowledge dir.
        while d != knowledge_dir_path.parent and d.exists():
            dirs_to_check.add(d)
            if d == knowledge_dir_path:
                break
            d = d.parent
    for d in sorted(dirs_to_check):
        if "_archive" in d.parts:
            continue
        if not (d / "README.md").exists():
            rel = d.relative_to(knowledge_dir_path) if d != knowledge_dir_path else Path(".")
            out.append(f"missing README.md in {rel}/")

    # Check 2: per-directory entry count.
    by_dir: Dict[Path, List[Path]] = {}
    for md in entry_files:
        by_dir.setdefault(md.parent, []).append(md)
    for d, mds in by_dir.items():
        if len(mds) > MAX_FLAT_DIR_ENTRIES:
            rel = d.relative_to(knowledge_dir_path) if d != knowledge_dir_path else Path(".")
            out.append(
                f"directory {rel}/ has {len(mds)} entry .md files "
                f"(max is {MAX_FLAT_DIR_ENTRIES})"
            )

    # Check 3: wikilink resolution.
    # Scan READMEs and index.md — those are the primary places real
    # navigation wikilinks live. Skip log.md: it's an audit trail that
    # quotes proposed-but-vetoed paths verbatim in veto reasons; treating
    # those quoted strings as wikilinks produces false positives.
    for md in all_md_files:
        if md.name == "log.md":
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in _WIKILINK_RE.finditer(text):
            link = m.group(1).split("|", 1)[0].strip()
            if not link:
                continue
            # Allow archive links (per AGENTS.md §1.2 those are intentional).
            if link.startswith("_archive/") or "/_archive/" in link:
                continue
            # Daily/transcript anchors are allowed (`[[daily/...]]`).
            if link.startswith("daily/"):
                continue
            # Folder-shaped links like [[daemon/]] resolve to that
            # folder's README.md.
            if link.endswith("/"):
                target = knowledge_dir_path / link / "README.md"
            else:
                target = knowledge_dir_path / (
                    link if link.endswith(".md") else f"{link}.md"
                )
            # Also tolerate links that are relative to the entry's own
            # directory rather than the knowledge root (e.g. a sibling
            # link in a README written as [[error-handling]] when the
            # README and the entry share a parent, or [[daemon/]] from
            # a parent README pointing at a child folder).
            if not target.exists():
                if link.endswith("/"):
                    sibling = md.parent / link / "README.md"
                else:
                    sibling = md.parent / (
                        link if link.endswith(".md") else f"{link}.md"
                    )
                if sibling.exists():
                    continue
                rel = md.relative_to(knowledge_dir_path)
                out.append(f"broken wikilink in {rel}: [[{link}]]")

    # Check 4: frontmatter validation + body present + scope agreement.
    for md in entry_files:
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = md.relative_to(knowledge_dir_path)
        fm = _parse_frontmatter(text)
        if not fm:
            out.append(f"missing or unparseable frontmatter in {rel}")
            continue
        missing = [f for f in _REQUIRED_FRONTMATTER_FIELDS if f not in fm]
        if missing:
            out.append(
                f"missing frontmatter fields in {rel}: {', '.join(missing)}"
            )

        # Body must contain something beyond the frontmatter block.
        body = _strip_frontmatter(text).strip()
        if len(body) < 20:
            out.append(f"entry body is empty or trivial in {rel}")

        # Scope/path agreement (prompt invariant #5).
        scope = str(fm.get("scope", "")).strip()
        parts = rel.parts
        if scope == "global":
            if parts[0] != "global":
                out.append(
                    f"scope/path mismatch in {rel}: scope=global but path "
                    f"is not under global/"
                )
        elif scope.startswith("project:"):
            slug = scope.split(":", 1)[1].strip()
            if parts[0] != "projects" or (len(parts) > 1 and parts[1] != slug):
                out.append(
                    f"scope/path mismatch in {rel}: scope={scope!r} but "
                    f"path is not under projects/{slug}/"
                )

    return out


def _strip_frontmatter(text: str) -> str:
    """Return ``text`` with a leading YAML frontmatter block removed."""
    m = _FRONTMATTER_HEAD_RE.match(text)
    if not m:
        return text
    return text[m.end():]


_FRONTMATTER_HEAD_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    m = _FRONTMATTER_HEAD_RE.match(text)
    if not m:
        return {}
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}
    return fm if isinstance(fm, dict) else {}


# ── README reconciliation (deterministic bookkeeping) ──────────────────
#
# After the Scholar finishes its action batch, walk every folder in the
# knowledge tree and make sure each README.md exists and its
# auto-managed "children list" section matches the folder's actual
# contents. The LLM owns prose; this owns the bullet list.
#
# Markers delimit the auto-managed block. Anything outside is preserved
# verbatim, so the Librarian can write a description above and trust the
# child-listing below to stay in sync.

_AUTO_CHILDREN_BEGIN = "<!-- ULTAN:children (auto) -->"
_AUTO_CHILDREN_END = "<!-- /ULTAN:children -->"


def _entry_title(entry: Path) -> str:
    """Return the entry's frontmatter title, falling back to its stem."""
    try:
        text = entry.read_text(encoding="utf-8")
    except OSError:
        return entry.stem
    fm = _parse_frontmatter(text)
    title = fm.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip().strip('"').strip("'")
    return entry.stem


def _folder_title(folder: Path) -> str:
    """Return the folder's README h1 if present, else humanise the name."""
    readme = folder / "README.md"
    if readme.exists():
        try:
            for line in readme.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
        except OSError:
            pass
    return folder.name.replace("-", " ").replace("_", " ").title()


def _format_children_section(folder: Path, knowledge_dir: Path) -> str:
    """Build the auto-managed children list for ``folder`` as text.

    Subfolders first (linked with trailing slash → README), then entries
    (linked with full path from knowledge root, no .md). Excludes
    ``_archive`` subtrees and the folder's own README.
    """
    lines = [_AUTO_CHILDREN_BEGIN]
    subfolders = sorted(
        p for p in folder.iterdir()
        if p.is_dir() and p.name != "_archive"
    )
    for sub in subfolders:
        rel = sub.relative_to(knowledge_dir)
        title = _folder_title(sub)
        lines.append(f"- [[{rel}/]] — {title}")
    entries = sorted(
        p for p in folder.glob("*.md")
        if p.name not in ("README.md", "index.md", "log.md")
    )
    for entry in entries:
        rel = entry.relative_to(knowledge_dir)
        link = str(rel.with_suffix(""))
        title = _entry_title(entry)
        lines.append(f"- [[{link}]] — {title}")
    if not subfolders and not entries:
        lines.append("_(empty)_")
    lines.append(_AUTO_CHILDREN_END)
    return "\n".join(lines)


def _splice_children_section(existing: str, section: str) -> str:
    """Replace the marker-delimited block in ``existing`` with ``section``.

    If the markers aren't present, append the section to the end with a
    blank-line separator. If only the BEGIN marker is present (corrupted
    file), append a fresh section and leave the corrupted one in place
    for the user to clean up — never silently destroy content.
    """
    begin = existing.find(_AUTO_CHILDREN_BEGIN)
    if begin == -1:
        sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        return existing + sep + section + "\n"
    end = existing.find(_AUTO_CHILDREN_END, begin)
    if end == -1:
        sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        return existing + sep + section + "\n"
    end += len(_AUTO_CHILDREN_END)
    return existing[:begin] + section + existing[end:]


def reconcile_readmes(knowledge_dir: Path) -> List[str]:
    """For every directory under ``knowledge_dir`` (excluding ``_archive``),
    ensure a README.md exists and its auto-managed children list matches
    the directory's actual contents.

    Returns a list of human-readable change messages. Empty list means
    everything was already in sync. Safe to run repeatedly; idempotent.

    This is the deterministic safety net for the rule "every folder has
    a README that reflects its contents." The Librarian/Scholar can
    still propose their own README updates (for prose); reconcile owns
    the bullet list of children below the auto markers.
    """
    if not knowledge_dir.exists():
        return []
    changes: List[str] = []

    # Collect every dir except _archive subtrees. Include the knowledge
    # root itself.
    all_dirs: List[Path] = [knowledge_dir]
    for d in knowledge_dir.rglob("*"):
        if d.is_dir() and "_archive" not in d.parts:
            all_dirs.append(d)

    for folder in sorted(set(all_dirs)):
        readme = folder / "README.md"
        section = _format_children_section(folder, knowledge_dir)
        if not readme.exists():
            display = (
                "Knowledge" if folder == knowledge_dir
                else folder.name.replace("-", " ").replace("_", " ").title()
            )
            body = f"# {display}\n\n{section}\n"
            try:
                readme.write_text(body, encoding="utf-8")
                changes.append(f"created README in {folder.relative_to(knowledge_dir) if folder != knowledge_dir else '.'}/")
            except OSError as e:
                log.warning("could not create %s: %s", readme, e)
            continue
        try:
            current = readme.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("could not read %s: %s", readme, e)
            continue
        updated = _splice_children_section(current, section)
        if updated != current:
            try:
                readme.write_text(updated, encoding="utf-8")
                rel = folder.relative_to(knowledge_dir) if folder != knowledge_dir else Path(".")
                changes.append(f"reconciled children in {rel}/README.md")
            except OSError as e:
                log.warning("could not write %s: %s", readme, e)

    return changes
