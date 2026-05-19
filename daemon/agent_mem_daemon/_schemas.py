"""Pydantic v2 models for Librarian and Scholar JSON responses.

The Librarian PROPOSES actions; the Scholar APPROVES or VETOES them.

- `LibrarianProposal` — top-level Librarian response. List of typed
  `ProposedAction` discriminated by ``action``. Plus optional `interrupts`
  (unchanged from the old design — the daemon still picks up nudge candidates).
- `ScholarReview` — top-level Scholar response. One `ScholarDecision` per
  proposed action (matched by ``action_index``). Decision is either
  ``approve`` or ``veto``; vetoes carry a one-sentence reason and DROP the
  proposal (no fix-ups — user-mandated).

Discipline:

- ``model_config = ConfigDict(extra="ignore")`` everywhere — small models
  add stray keys; we tolerate them.
- Every field on the action models is permitted to be missing/None at
  validation time (the Librarian sometimes emits partial JSON). The
  Scholar's veto path catches missing-but-required fields downstream.
- The discriminated union is keyed on ``action`` (Pydantic's
  ``Field(discriminator=...)``). Unknown action strings are rejected at
  parse time so the Scholar never sees a phantom action type.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


# ── Common building blocks (kept for backward compat: the Librarian's
# interrupt path is unchanged and tests for the response_parser may still
# import these) ────────────────────────────────────────────────────────


class EvidenceItem(BaseModel):
    """One quoted-evidence line: (turn_id, role, quote)."""

    model_config = ConfigDict(extra="ignore")

    turn_id: Optional[int] = None
    role: Optional[str] = None
    quote: Optional[str] = None


class BM25Hit(BaseModel):
    """One BM25 near-hit attached to a candidate."""

    model_config = ConfigDict(extra="ignore")

    entry_id: Optional[str] = None
    score: Optional[float] = None
    path: Optional[str] = None


# ── ProposedAction discriminated union ───────────────────────────────


class _BaseAction(BaseModel):
    """Common shape across every proposed action."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    """One short paragraph explaining why this action is proposed.
    The Scholar reads this to decide approve/veto."""


class WriteEntry(_BaseAction):
    """Create a new entry at ``path`` with ``body``.

    ``path`` is relative to ``knowledge/`` (e.g.
    ``global/tooling/uv-basics.md``). The Scholar checks the directory
    is well-formed (README exists, capacity not exceeded) before
    approving."""

    action: Literal["write_entry"] = "write_entry"
    path: str = ""
    body: str = ""


class UpdateEntry(_BaseAction):
    """Replace the contents of an existing entry at ``path`` with
    ``new_body`` (full file body, frontmatter included)."""

    action: Literal["update_entry"] = "update_entry"
    path: str = ""
    new_body: str = ""


class MergeEntries(_BaseAction):
    """Combine N entries into one.

    ``source_paths`` is moved to ``_archive/`` (preserving structure).
    ``target_path`` is created (or overwritten) with ``target_body``.
    If ``target_path`` is one of ``source_paths``, the others are
    archived and the target is overwritten in place.
    """

    action: Literal["merge_entries"] = "merge_entries"
    source_paths: List[str] = Field(default_factory=list)
    target_path: str = ""
    target_body: str = ""


class MoveEntry(_BaseAction):
    """Move an entry from ``from_path`` to ``to_path``.

    Contents are preserved verbatim except the ``id`` frontmatter field,
    which the Scholar rewrites to match the new filename. Wikilinks
    pointing at the old path are NOT auto-updated — leave that for a
    future deterministic pass."""

    action: Literal["move_entry"] = "move_entry"
    from_path: str = ""
    to_path: str = ""


class ArchiveEntry(_BaseAction):
    """Move ``path`` to ``_archive/`` (preserving subdirectory structure).

    The archived entry keeps its frontmatter but gets ``status: stale``
    appended if not already present, plus an ``archived: <today>``
    field for traceability."""

    action: Literal["archive_entry"] = "archive_entry"
    path: str = ""


class DeprecateEntry(_BaseAction):
    """Mark ``path`` as deprecated in favour of a replacement.

    Use this — not ``archive_entry`` — when the user has changed their
    mind and a newer entry now covers the same topic better. Effects:

      - Sets ``status: deprecated`` in the frontmatter (keeps the file
        in place so wikilinks pointing at it still resolve).
      - Adds a ``superseded_by: <replacement_path>`` frontmatter field.
      - Appends a "**Superseded by [[<replacement_path>]] as of <today>**"
        note at the top of the body so a reader who lands here knows
        where to look.

    Deprecated entries stay in folder listings but a future search /
    advisor pass should prefer the replacement. Archive only when the
    entry is wrong, irrelevant, or has no successor."""

    action: Literal["deprecate_entry"] = "deprecate_entry"
    path: str = ""
    superseded_by: str = ""


class UpdateReadme(_BaseAction):
    """Rewrite the README.md inside ``folder_path`` with ``new_body``.

    ``folder_path`` is relative to ``knowledge/`` (e.g. ``global/tooling``
    or ``""`` for the root README)."""

    action: Literal["update_readme"] = "update_readme"
    folder_path: str = ""
    new_body: str = ""


class AddWikilink(_BaseAction):
    """Add a ``[[wikilink]]`` from ``from_path`` to ``to_path``.

    The Scholar inserts a Related section bullet (or adds to the
    existing one). ``context`` is one sentence describing why the link
    is relevant (rendered alongside the link itself)."""

    action: Literal["add_wikilink"] = "add_wikilink"
    from_path: str = ""
    to_path: str = ""
    context: str = ""


class SplitFolder(_BaseAction):
    """Restructure ``folder_path`` into subfolders.

    ``into`` maps subfolder name → list of entry paths (relative to
    ``knowledge/``) that should move there. Entries not listed stay
    where they are. The Scholar creates README.md for each new
    subfolder."""

    action: Literal["split_folder"] = "split_folder"
    folder_path: str = ""
    into: Dict[str, List[str]] = Field(default_factory=dict)


# Discriminated union — Pydantic dispatches on the ``action`` literal.
ProposedAction = Annotated[
    Union[
        WriteEntry,
        UpdateEntry,
        MergeEntries,
        MoveEntry,
        ArchiveEntry,
        DeprecateEntry,
        UpdateReadme,
        AddWikilink,
        SplitFolder,
    ],
    Field(discriminator="action"),
]


# ── Librarian top-level response ─────────────────────────────────────


class LibrarianInterrupt(BaseModel):
    """A single interrupt candidate (unchanged from the previous design).

    The interrupt-nudge path is orthogonal to the curator restructure.
    The Librarian still emits these when a turn matches an existing
    confirmed entry's applies-when phrases; the Scholar still routes
    approvals into ``pending-nudges.md``.
    """

    model_config = ConfigDict(extra="ignore")

    lesson_id: Optional[str] = None
    lesson_path: Optional[str] = None
    matching_applies_when: Optional[str] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)
    match_score: Optional[float] = None
    librarian_confidence: Optional[float] = None


class LibrarianProposal(BaseModel):
    """Top-level Librarian response in the new architecture.

    ``proposals`` is the ordered list of actions the Librarian wants
    executed. Order matters: the Scholar processes them in order and may
    veto any of them. ``interrupts`` is unchanged and optional.
    """

    model_config = ConfigDict(extra="ignore")

    proposals: List[ProposedAction] = Field(default_factory=list)
    interrupts: List[LibrarianInterrupt] = Field(default_factory=list)


# ── Scholar review ───────────────────────────────────────────────────


class ScholarDecision(BaseModel):
    """One decision in the Scholar's review.

    ``action_index`` is the 0-based position of the corresponding
    proposal in ``LibrarianProposal.proposals``. ``decision`` is either
    ``approve`` or ``veto``. ``veto_reason`` is required on veto and
    ignored on approve.
    """

    model_config = ConfigDict(extra="ignore")

    action_index: int = -1
    decision: Literal["approve", "veto"] = "veto"
    veto_reason: str = ""


class ScholarInterruptDecision(BaseModel):
    """Mirror of the old ``ScholarInterruptDecision``.

    Kept identical so the nudge file path (used by the hook layer)
    continues to work without changes.
    """

    model_config = ConfigDict(extra="ignore")

    lesson_id: Optional[str] = None
    lesson_path: Optional[str] = None
    action: Optional[str] = None  # "approve" | "veto"
    text: Optional[str] = None
    reason: Optional[str] = None


class ScholarReview(BaseModel):
    """Top-level Scholar response in the new architecture.

    ``decisions`` mirrors the order of the Librarian's proposals
    (matched by ``action_index``). ``interrupts_processed`` mirrors the
    old shape so the nudge pipeline stays intact.
    """

    model_config = ConfigDict(extra="ignore")

    decisions: List[ScholarDecision] = Field(default_factory=list)
    interrupts_processed: List[ScholarInterruptDecision] = Field(default_factory=list)


# ── Backwards-compat aliases (kept so test_response_parser and any
# leftover references still type-check) ──────────────────────────────


class LibrarianResponse(BaseModel):
    """Legacy alias — accepts the old shape AND the new shape.

    A handful of callers (and one response_parser test path) still
    reference this name. The model is permissive: it accepts either the
    legacy ``candidates``/``interrupt_candidates`` shape OR the new
    ``proposals``/``interrupts`` shape. Validation never fails on shape;
    callers must inspect the populated lists.
    """

    model_config = ConfigDict(extra="ignore")

    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    interrupt_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    proposals: List[Dict[str, Any]] = Field(default_factory=list)
    interrupts: List[Dict[str, Any]] = Field(default_factory=list)


class ScholarResponse(BaseModel):
    """Legacy alias for the old Scholar shape.

    Still accepted by the parser for nudge-pipeline backward compat.
    """

    model_config = ConfigDict(extra="ignore")

    candidates_processed: List[Dict[str, Any]] = Field(default_factory=list)
    interrupts_processed: List[Dict[str, Any]] = Field(default_factory=list)


# ── Prompt-shape generators ──────────────────────────────────────────
#
# Single source of truth: the Pydantic models above ARE the schema. The
# prompt instructions describing the JSON the LLM should produce are
# generated from those models — never hand-written and copy-pasted —
# so they can't drift.
#
# Two helpers:
#   - ``describe_action_types_markdown()`` for the Librarian's
#     ProposedAction enumeration.
#   - ``describe_librarian_response_shape()`` / ``describe_scholar_response_shape()``
#     for the JSON envelope each role must emit.


_ACTION_CLASSES = [
    WriteEntry,
    UpdateEntry,
    MergeEntries,
    MoveEntry,
    ArchiveEntry,
    DeprecateEntry,
    UpdateReadme,
    AddWikilink,
    SplitFolder,
]


def _action_name(cls: type[BaseModel]) -> str:
    """Pull the ``action`` literal value off an action class.

    Pydantic stores the default on ``model_fields["action"].default``.
    """
    field = cls.model_fields.get("action")
    return str(field.default) if field is not None else cls.__name__.lower()


def _action_payload_fields(cls: type[BaseModel]) -> List[str]:
    """Non-discriminator, non-base field names for an action class, in
    declaration order. Excludes ``action`` (the discriminator) and
    ``reasoning`` (inherited; documented once globally)."""
    skip = {"action", "reasoning"}
    return [name for name in cls.model_fields if name not in skip]


def _first_sentence(text: str) -> str:
    """Return the first sentence of a docstring (split on blank line or
    period+space). Trims whitespace and trailing periods for prompt fit.
    """
    if not text:
        return ""
    flat = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    for sep in (". ", ".\n", ".\t"):
        if sep in flat:
            return flat.split(sep, 1)[0].strip()
    return flat.rstrip(".").strip()


def describe_action_types_markdown() -> str:
    """Render the ProposedAction enumeration as a Markdown block ready
    to inline in the Librarian prompt. Always reflects the current
    Pydantic schema — change the model, change the prompt.
    """
    lines: List[str] = []
    for cls in _ACTION_CLASSES:
        name = _action_name(cls)
        fields = _action_payload_fields(cls) + ["reasoning"]
        summary = _first_sentence(cls.__doc__ or "")
        # Right-pad the action name so the descriptions line up cleanly
        # when the model reads them.
        lines.append(f"  {name:<16} — {summary}. Fields: {', '.join(fields)}.")
    return "\n".join(lines)


def describe_librarian_response_shape() -> str:
    """JSON skeleton the Librarian must emit. Derived from
    ``LibrarianProposal``/``LibrarianInterrupt`` + the action shapes
    above. Returns a fenced ``json`` block as plain text.
    """
    return (
        '```json\n'
        '{\n'
        '  "proposals": [\n'
        '    {"action": "<one of: '
        + ", ".join(_action_name(c) for c in _ACTION_CLASSES)
        + '>", "reasoning": "<one sentence>", "<action-specific fields...>": "..."},\n'
        '    ...\n'
        '  ],\n'
        '  "interrupts": [\n'
        '    {"lesson_id": "...", "lesson_path": "...", '
        '"matching_applies_when": "...", "evidence": [...], '
        '"match_score": 0.0, "librarian_confidence": 0.0},\n'
        '    ...\n'
        '  ]\n'
        '}\n'
        '```'
    )


def describe_scholar_response_shape() -> str:
    """JSON skeleton the Scholar must emit. Derived from
    ``ScholarReview``/``ScholarDecision``/``ScholarInterruptDecision``.
    """
    return (
        '```json\n'
        '{\n'
        '  "decisions": [\n'
        '    {"action_index": 0, "decision": "approve|veto", '
        '"veto_reason": "<sentence if veto, else empty>"},\n'
        '    ...\n'
        '  ],\n'
        '  "interrupts_processed": [\n'
        '    {"lesson_id": "...", "lesson_path": "...", '
        '"action": "approve|veto", "text": "<nudge if approve>", '
        '"reason": "<short>"},\n'
        '    ...\n'
        '  ]\n'
        '}\n'
        '```'
    )
