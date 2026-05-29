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

import json
from typing import Dict, List, Literal, Mapping, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated

from . import _validation

# ── Common building blocks (kept for backward compat: the Librarian's
# interrupt path is unchanged and tests for the response_parser may still
# import these) ────────────────────────────────────────────────────────


class EvidenceItem(BaseModel):
    """One quoted-evidence line — a single turn the Librarian wants
    the Scholar to look at when judging an interrupt."""

    model_config = ConfigDict(extra="ignore")

    turn_id: Optional[int] = Field(
        default=None,
        description="1-based turn id from the rolling buffer.",
    )
    role: Optional[str] = Field(
        default=None,
        description='"user" or "assistant".',
    )
    quote: Optional[str] = Field(
        default=None,
        description="Verbatim text snippet from the turn.",
    )


class BM25Hit(BaseModel):
    """One BM25 near-hit attached to a candidate."""

    model_config = ConfigDict(extra="ignore")

    entry_id: Optional[str] = Field(default=None, description="Frontmatter `id` of the hit.")
    score: Optional[float] = Field(default=None, description="BM25 score.")
    path: Optional[str] = Field(default=None, description="Path relative to knowledge/.")


# ── ProposedAction discriminated union ───────────────────────────────


class _BaseAction(BaseModel):
    """Common shape across every proposed action."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = Field(
        default="",
        description=(
            "One short paragraph explaining why this action is "
            "proposed. The Scholar reads this to decide approve/veto."
        ),
    )

    salience_signal: Optional[Literal["contradicts", "novel", "reinforces", "used_helpfully"]] = (
        Field(
            default=None,
            description=(
                "Why this is worth remembering, in cognitive-science "
                "terms. 'contradicts': disagrees with an existing entry "
                "(cite path in existing_entry); user has changed their "
                "mind. 'novel': not in library AND not derivable from "
                "the model's baseline knowledge. 'reinforces': restates "
                "an existing entry's claim (cite path in existing_entry); "
                "the daemon will bump that entry's reinforced counter "
                "and the Scholar may veto the write. 'used_helpfully': "
                "the assistant actually RELIED ON / AGREED WITH a "
                "surfaced or cited existing entry to answer THIS turn "
                "(cite path in existing_entry AND the stable turn id in "
                "cited_turn_seq); the daemon bumps that entry's "
                "fired-helpful counter once per cited turn. A mere "
                "mention is NOT use, and disagreement is 'contradicts', "
                "not this. Null if unsure — the Scholar will infer."
            ),
        )
    )

    existing_entry: Optional[str] = Field(
        default=None,
        description=(
            "For 'contradicts'/'reinforces'/'used_helpfully' signals: "
            "path of the existing library entry the candidate relates "
            "to, relative to knowledge/. Required for those signals so "
            "the Scholar can verify the relationship."
        ),
    )

    cited_turn_seq: Optional[int] = Field(
        default=None,
        description=(
            "For the 'used_helpfully' signal ONLY: the STABLE turn id "
            "(the `(turn_seq=S)` token rendered on the rolling-buffer "
            "line — NOT the bracketed `[N]` scan-local label) of the "
            "turn in which the assistant relied on `existing_entry`. The "
            "daemon dedups fired-helpful bumps on (session, entry, "
            "turn_seq) so re-seeing the same turn on a later scan does "
            "not double-count. Required for 'used_helpfully'; ignored "
            "for other signals."
        ),
    )


class WriteEntry(_BaseAction):
    """Create a new entry at ``path`` with ``body``.

    ``path`` is relative to ``knowledge/`` (e.g.
    ``global/tooling/uv-basics.md``). The Scholar checks the directory
    is well-formed (README exists, capacity not exceeded) before
    approving."""

    action: Literal["write_entry"] = "write_entry"
    path: str = Field(
        default="",
        description="Target path for the new entry, relative to knowledge/.",
    )
    body: str = Field(default="", description="Full markdown body including frontmatter.")


class UpdateEntry(_BaseAction):
    """Replace the contents of an existing entry at ``path`` with
    ``new_body`` (full file body, frontmatter included)."""

    action: Literal["update_entry"] = "update_entry"
    path: str = Field(default="", description="Existing entry path, relative to knowledge/.")
    new_body: str = Field(
        default="",
        description="Full replacement markdown body (frontmatter + body).",
    )


class MergeEntries(_BaseAction):
    """Combine N entries into one.

    ``source_paths`` is moved to ``_archive/`` (preserving structure).
    ``target_path`` is created (or overwritten) with ``target_body``.
    If ``target_path`` is one of ``source_paths``, the others are
    archived and the target is overwritten in place.
    """

    action: Literal["merge_entries"] = "merge_entries"
    source_paths: List[str] = Field(
        default_factory=list,
        description=("List of existing entry paths (relative to knowledge/) to merge and archive."),
    )
    target_path: str = Field(default="", description="Where the merged entry should live.")
    target_body: str = Field(default="", description="Full markdown body of the merged result.")


class MoveEntry(_BaseAction):
    """Move an entry from ``from_path`` to ``to_path``.

    Contents are preserved verbatim except the ``id`` frontmatter field,
    which the Scholar rewrites to match the new filename. Wikilinks
    pointing at the old path are NOT auto-updated — leave that for a
    future deterministic pass."""

    action: Literal["move_entry"] = "move_entry"
    from_path: str = Field(default="", description="Existing path, relative to knowledge/.")
    to_path: str = Field(default="", description="New path, relative to knowledge/.")


class ArchiveEntry(_BaseAction):
    """Move ``path`` to ``_archive/`` (preserving subdirectory structure).

    The archived entry keeps its frontmatter but gets ``status: stale``
    appended if not already present, plus an ``archived: <today>``
    field for traceability."""

    action: Literal["archive_entry"] = "archive_entry"
    path: str = Field(default="", description="Entry path to archive, relative to knowledge/.")


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
    path: str = Field(
        default="",
        description="Older entry to deprecate, relative to knowledge/.",
    )
    superseded_by: str = Field(
        default="",
        description=("Newer entry path that supersedes this one, relative to knowledge/."),
    )


class UpdateReadme(_BaseAction):
    """Rewrite the README.md inside ``folder_path`` with ``new_body``.

    ``folder_path`` is relative to ``knowledge/`` (e.g. ``global/tooling``
    or ``""`` for the root README)."""

    action: Literal["update_readme"] = "update_readme"
    folder_path: str = Field(
        default="",
        description=('Folder path relative to knowledge/, or "" for the root README.'),
    )
    new_body: str = Field(
        default="",
        description=(
            "Prose only — daemon auto-maintains the child listing below the marker comments."
        ),
    )


class AddWikilink(_BaseAction):
    """Add a ``[[wikilink]]`` from ``from_path`` to ``to_path``.

    The Scholar inserts a Related section bullet (or adds to the
    existing one). ``context`` is one sentence describing why the link
    is relevant (rendered alongside the link itself)."""

    action: Literal["add_wikilink"] = "add_wikilink"
    from_path: str = Field(
        default="",
        description=("Entry whose Related section gets the new link (relative to knowledge/)."),
    )
    to_path: str = Field(
        default="",
        description="Target entry path (relative to knowledge/).",
    )
    context: str = Field(default="", description="One-sentence why the link is relevant.")


class SplitFolder(_BaseAction):
    """Restructure ``folder_path`` into subfolders.

    ``into`` maps subfolder name → list of entry paths (relative to
    ``knowledge/``) that should move there. Entries not listed stay
    where they are. The Scholar creates README.md for each new
    subfolder."""

    action: Literal["split_folder"] = "split_folder"
    folder_path: str = Field(
        default="",
        description="Folder to restructure, relative to knowledge/.",
    )
    into: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of new subfolder name → list of entry paths "
            "(relative to knowledge/) to move into it."
        ),
    )


class AbstractEntries(_BaseAction):
    """Synthesise a higher-order PARENT abstraction over ≥2 related leaf
    entries (reflective abstraction — bottom-up schema formation).

    The children STAY in place (each remains individually retrievable and
    subject to normal decay); a new parent entry is created at
    ``parent_path`` whose ``parent_body`` states a single general rule
    abstracted FROM them and links to each via ``[[wikilink]]``. The
    executor also writes a reverse ``[[parent]]`` backlink into each child.

    This is NOT dedup (``merge_entries`` archives its sources) and NOT
    reorg (``split_folder``/``move_entry`` relocate without synthesising):
    nothing is archived or moved. Propose ONLY for a genuine "aha" — a
    surprising, non-obvious, PREDICTIVE rule that connects leaves from
    DIFFERENT domains and would let a competent assistant make a confident
    call on an unseen case no single child supports. ``parent_body`` is
    full markdown including YAML frontmatter and MUST use ``type:
    abstraction``."""

    action: Literal["abstract_entries"] = "abstract_entries"
    child_paths: List[str] = Field(
        default_factory=list,
        description=(
            "≥2 existing leaf entry paths (relative to knowledge/) the "
            "parent abstracts. Must contain at least two; an abstraction "
            "over a single entry is not an abstraction."
        ),
    )
    parent_path: str = Field(
        default="",
        description=(
            "Where the synthesised parent entry should live, relative to knowledge/, ending in .md."
        ),
    )
    parent_title: str = Field(
        default="",
        description="Human-readable title of the parent abstraction.",
    )
    parent_body: str = Field(
        default="",
        description=(
            "Full markdown body of the parent entry, including YAML "
            "frontmatter with ``type: abstraction`` and ``[[wikilink]]``s "
            "to each child."
        ),
    )

    @model_validator(mode="after")
    def _require_two_children(self) -> "AbstractEntries":
        """An abstraction needs ≥2 children. The Librarian sometimes emits
        partial JSON, but a single-child (or empty) ``child_paths`` is never
        a valid abstraction — reject it at parse time so neither the Scholar
        nor the executor sees a degenerate action."""
        if len(self.child_paths) < 2:
            raise ValueError(
                "abstract_entries requires at least two child_paths "
                "(an abstraction over fewer than two leaves is not an abstraction)"
            )
        return self


# The bare union of proposed-action classes — usable as a type annotation
# and for ``isinstance`` in the Librarian's boundary validator. The
# discriminated alias below adds Pydantic's ``action`` dispatch on top.
ProposedActionT = Union[
    WriteEntry,
    UpdateEntry,
    MergeEntries,
    MoveEntry,
    ArchiveEntry,
    DeprecateEntry,
    UpdateReadme,
    AddWikilink,
    SplitFolder,
    AbstractEntries,
]

# Discriminated union — Pydantic dispatches on the ``action`` literal.
ProposedAction = Annotated[
    ProposedActionT,
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

    lesson_id: Optional[str] = Field(
        default=None,
        description="Frontmatter `id` from the applies-when table.",
    )
    lesson_path: Optional[str] = Field(
        default=None,
        description="Path of the matched lesson, relative to knowledge/.",
    )
    matching_applies_when: Optional[str] = Field(
        default=None,
        description="The exact applies-when phrase that matched the buffer.",
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list[EvidenceItem],
        description=(
            "List of EvidenceItem dicts (NOT strings). Each item is "
            "{turn_id, role, quote}. Emit at least one verbatim turn quote."
        ),
    )
    match_score: Optional[float] = Field(
        default=None,
        description=("0.0 to 1.0 — how strongly the buffer matched the applies-when phrase."),
    )
    librarian_confidence: Optional[float] = Field(
        default=None,
        description=("0.0 to 1.0 — how confident the Librarian is this is worth interrupting on."),
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: object) -> object:
        """Tolerate LLM-shape drift on evidence.

        The schema asks for ``list[EvidenceItem]`` (dicts), but Sonnet /
        Haiku sometimes emit:
          - a single string (the full quoted turn)
          - a list of strings
          - a single dict instead of a list
        Coerce all of those to the canonical list-of-dicts shape so we
        don't drop the entire interrupt over a formatting variation.
        Validation errors on individual items still surface in the
        normal way.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [{"quote": v}]
        if isinstance(v, Mapping):
            return [cast(Mapping[object, object], v)]
        if isinstance(v, list):
            raw = cast(list[object], v)
            out: list[object] = []
            for item in raw:
                if isinstance(item, str):
                    out.append({"quote": item})
                else:
                    out.append(item)
            return out
        return v


class LibrarianProposal(BaseModel):
    """Top-level Librarian response in the new architecture.

    ``proposals`` is the ordered list of actions the Librarian wants
    executed. Order matters: the Scholar processes them in order and may
    veto any of them. ``interrupts`` is unchanged and optional.
    """

    model_config = ConfigDict(extra="ignore")

    proposals: List[ProposedAction] = Field(
        default_factory=list[ProposedAction],
        description=(
            "Ordered list of curator actions the Librarian wants the "
            "Scholar to execute. Each item is a typed action object "
            "discriminated by `action` (see the action enumeration). "
            "Order matters — the Scholar processes them sequentially "
            "and any may be vetoed. Emit [] when no curation is needed."
        ),
    )
    interrupts: List[LibrarianInterrupt] = Field(
        default_factory=list[LibrarianInterrupt],
        description=(
            "Optional list of interrupt-nudge candidates: turns in the "
            "buffer that match an existing entry's applies-when phrases. "
            "Independent of `proposals`. Emit [] when nothing matched."
        ),
    )


# ── Scholar review ───────────────────────────────────────────────────


class ScholarDecision(BaseModel):
    """One decision in the Scholar's review.

    ``action_index`` is the 0-based position of the corresponding
    proposal in ``LibrarianProposal.proposals``. ``decision`` is either
    ``approve`` or ``veto``. ``veto_reason`` is required on veto and
    ignored on approve.
    """

    model_config = ConfigDict(extra="ignore")

    action_index: int = Field(
        default=-1,
        description=(
            "0-based index of the proposal in "
            "`LibrarianProposal.proposals` this decision applies to. "
            "Must be a valid index; mismatched indices are dropped."
        ),
    )
    decision: Literal["approve", "veto"] = Field(
        default="veto",
        description=(
            "Either 'approve' (the daemon executes the action) or "
            "'veto' (the action is dropped — no fix-ups). Default "
            "'veto' is fail-safe: an unparseable decision drops the "
            "proposal rather than executing it."
        ),
    )
    veto_reason: str = Field(
        default="",
        description=(
            "Required on veto: one short sentence explaining why the "
            "proposal was rejected. Surfaced in run logs for debugging. "
            "Ignored on approve."
        ),
    )


class ScholarInterruptDecision(BaseModel):
    """Mirror of the old ``ScholarInterruptDecision``.

    Kept identical so the nudge file path (used by the hook layer)
    continues to work without changes.
    """

    model_config = ConfigDict(extra="ignore")

    lesson_id: Optional[str] = Field(
        default=None,
        description="Frontmatter `id` of the lesson the interrupt was about.",
    )
    lesson_path: Optional[str] = Field(
        default=None,
        description="Path of the matched lesson, relative to knowledge/.",
    )
    action: Optional[str] = Field(
        default=None,
        description=(
            "Either 'approve' (the nudge is written to pending-nudges.md "
            "and surfaced to the user on the next prompt) or 'veto' "
            "(the candidate is dropped silently)."
        ),
    )
    text: Optional[str] = Field(
        default=None,
        description=(
            "On approve: the exact nudge text to surface to the user. "
            "One short sentence framed as a reminder of the relevant lesson."
        ),
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "On veto: one short sentence explaining why the interrupt "
            "was not worth surfacing. Ignored on approve."
        ),
    )


class ScholarReview(BaseModel):
    """Top-level Scholar response in the new architecture.

    ``decisions`` mirrors the order of the Librarian's proposals
    (matched by ``action_index``). ``interrupts_processed`` mirrors the
    old shape so the nudge pipeline stays intact.
    """

    model_config = ConfigDict(extra="ignore")

    decisions: List[ScholarDecision] = Field(
        default_factory=list[ScholarDecision],
        description=(
            "One ScholarDecision per Librarian proposal, matched by "
            "`action_index`. Order need not mirror the proposals — the "
            "daemon dispatches by `action_index`. Missing indices are "
            "treated as implicit vetoes."
        ),
    )
    interrupts_processed: List[ScholarInterruptDecision] = Field(
        default_factory=list[ScholarInterruptDecision],
        description=(
            "One ScholarInterruptDecision per LibrarianInterrupt the "
            "Scholar reviewed. Approved decisions are written to "
            "pending-nudges.md; vetoed decisions are dropped."
        ),
    )


# ── Scholar typed-action output (Pydantic AI migration) ──────────────
#
# In the Pydantic-AI architecture the Scholar no longer hand-writes files
# nor emits an approve/veto ``ScholarReview``. Instead it RETURNS the
# structural actions it has decided to apply, as a typed
# ``ScholarDecisions`` object, and the daemon's deterministic executor
# applies them. The action shapes mirror the Librarian's ``ProposedAction``
# union one-for-one, but these subclasses add boundary VALIDATORS: the
# Scholar's emitted actions must be well-formed (frontmatter parses, id
# matches the filename slug, scope agrees with the path) or Pydantic AI
# bounces them back to the model via ``ModelRetry``.
#
# The Librarian's loose ``ProposedAction`` union (above) is deliberately
# left untouched — the Librarian proposes partial JSON and the Scholar is
# its gatekeeper; only the Scholar's *output* is held to the strict bar.


def _frontmatter_field_errors(body: str, *, path: str) -> List[str]:
    """Self-contained frontmatter checks shared by every body-carrying
    Scholar action. Returns a list of human-readable error strings (empty
    when the body is well-formed). Skips silently when ``body`` is empty —
    an empty body is caught by the action's own field requirements, not
    here."""
    errors: List[str] = []
    if not body.strip():
        return errors
    fm = _validation.parse_frontmatter(body)
    if not fm:
        errors.append(
            f"body for {path!r} has no parseable YAML frontmatter block "
            "(expected a leading '---' ... '---' section)"
        )
        return errors
    missing = _validation.missing_frontmatter_fields(fm)
    if missing:
        errors.append(f"frontmatter for {path!r} is missing required fields: {', '.join(missing)}")
    if path:
        expected_id = _validation.path_slug(path)
        actual_id = str(fm.get("id", "")).strip()
        if actual_id and actual_id != expected_id:
            errors.append(
                f"frontmatter id {actual_id!r} does not match the filename slug "
                f"{expected_id!r} for path {path!r}"
            )
        scope = str(fm.get("scope", "")).strip()
        if scope:
            from pathlib import Path  # noqa: PLC0415 — local, validator-only

            scope_err = _validation.scope_path_violation(Path(path), scope)
            if scope_err:
                errors.append(scope_err)
    return errors


class ScholarWriteEntry(WriteEntry):
    """Scholar-validated ``write_entry`` — body frontmatter must parse, carry
    the required fields, and agree (id/scope) with ``path``."""

    @model_validator(mode="after")
    def _validate_body(self) -> "ScholarWriteEntry":
        errors = _frontmatter_field_errors(self.body, path=self.path)
        if not self.path.strip():
            errors.append("write_entry requires a non-empty `path`")
        if not self.body.strip():
            errors.append(f"write_entry for {self.path!r} requires a non-empty `body`")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class ScholarUpdateEntry(UpdateEntry):
    """Scholar-validated ``update_entry`` — ``new_body`` held to the same
    frontmatter bar as a fresh write."""

    @model_validator(mode="after")
    def _validate_body(self) -> "ScholarUpdateEntry":
        errors = _frontmatter_field_errors(self.new_body, path=self.path)
        if not self.path.strip():
            errors.append("update_entry requires a non-empty `path`")
        if not self.new_body.strip():
            errors.append(f"update_entry for {self.path!r} requires a non-empty `new_body`")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class ScholarMergeEntries(MergeEntries):
    """Scholar-validated ``merge_entries`` — ``target_body`` held to the
    frontmatter bar against ``target_path``."""

    @model_validator(mode="after")
    def _validate_body(self) -> "ScholarMergeEntries":
        errors = _frontmatter_field_errors(self.target_body, path=self.target_path)
        if not self.target_path.strip():
            errors.append("merge_entries requires a non-empty `target_path`")
        if not self.target_body.strip():
            errors.append(f"merge_entries for {self.target_path!r} requires a non-empty body")
        if not self.source_paths:
            errors.append("merge_entries requires at least one `source_paths` entry")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class ScholarMoveEntry(MoveEntry):
    """Scholar-validated ``move_entry`` — both endpoints required."""

    @model_validator(mode="after")
    def _validate_paths(self) -> "ScholarMoveEntry":
        if not self.from_path.strip() or not self.to_path.strip():
            raise ValueError("move_entry requires non-empty `from_path` and `to_path`")
        return self


class ScholarArchiveEntry(ArchiveEntry):
    """Scholar-validated ``archive_entry`` — ``path`` required."""

    @model_validator(mode="after")
    def _validate_path(self) -> "ScholarArchiveEntry":
        if not self.path.strip():
            raise ValueError("archive_entry requires a non-empty `path`")
        return self


class ScholarDeprecateEntry(DeprecateEntry):
    """Scholar-validated ``deprecate_entry`` — both ``path`` and
    ``superseded_by`` required (a deprecation with no successor should be an
    archive instead)."""

    @model_validator(mode="after")
    def _validate_paths(self) -> "ScholarDeprecateEntry":
        if not self.path.strip() or not self.superseded_by.strip():
            raise ValueError("deprecate_entry requires non-empty `path` and `superseded_by`")
        return self


class ScholarAbstractEntries(AbstractEntries):
    """Scholar-validated ``abstract_entries`` — ``parent_body`` held to the
    same frontmatter bar as a fresh write (parseable, required fields, id
    matches the filename slug, scope agrees with the path) against
    ``parent_path``. The ≥2-children rule is inherited from
    ``AbstractEntries``; per-child existence is a whole-batch check the
    executor cannot do in isolation, so it lives in ``scholar`` /
    ``librarian`` validators that see the knowledge root, not here."""

    @model_validator(mode="after")
    def _validate_body(self) -> "ScholarAbstractEntries":
        errors = _frontmatter_field_errors(self.parent_body, path=self.parent_path)
        if not self.parent_path.strip():
            errors.append("abstract_entries requires a non-empty `parent_path`")
        if not self.parent_body.strip():
            errors.append(
                f"abstract_entries for {self.parent_path!r} requires a non-empty `parent_body`"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self


# Discriminated union of the Scholar's validated actions. Same
# ``action`` discriminator keys as ``ProposedAction`` so the prompt
# enumeration generated from those models still matches.
ScholarAction = Annotated[
    Union[
        ScholarWriteEntry,
        ScholarUpdateEntry,
        ScholarMergeEntries,
        ScholarMoveEntry,
        ScholarArchiveEntry,
        ScholarDeprecateEntry,
        ScholarAbstractEntries,
    ],
    Field(discriminator="action"),
]


class ScholarDecisions(BaseModel):
    """Top-level Scholar output in the Pydantic-AI architecture.

    ``actions`` is the ordered list of structural changes the Scholar has
    APPROVED and wants the daemon to apply (vetoed proposals simply do not
    appear). ``interrupts_processed`` keeps the nudge pipeline intact and
    is identical to the old ``ScholarReview`` field.

    The daemon's executor walks ``actions`` in order and applies each to
    disk; the Pydantic field/model validators above plus the agent's
    ``output_validator`` guarantee each action is well-formed before the
    executor runs.
    """

    model_config = ConfigDict(extra="ignore")

    actions: List[ScholarAction] = Field(
        default_factory=list[ScholarAction],
        description=(
            "Ordered list of structural actions the Scholar has APPROVED "
            "and the daemon will apply, each a typed object discriminated "
            "by `action`. Emit [] when every proposal is vetoed. Each "
            "write/update/merge body MUST include valid YAML frontmatter "
            "whose `id` matches the filename slug and whose `scope` agrees "
            "with the path. Every [[wikilink]] in a body must resolve to an "
            "existing entry or to a path another action in this same list "
            "creates."
        ),
    )
    interrupts_processed: List[ScholarInterruptDecision] = Field(
        default_factory=list[ScholarInterruptDecision],
        description=(
            "One ScholarInterruptDecision per LibrarianInterrupt the Scholar "
            "reviewed. Approved decisions are written to pending-nudges.md; "
            "vetoed decisions are dropped."
        ),
    )


def describe_scholar_decisions_shape() -> str:
    """Canonical JSON Schema for ``ScholarDecisions`` derived from the
    Pydantic model. Inlined into the Scholar prompt so its description of
    the output can never drift from the model the agent validates against."""
    return _model_schema_block(ScholarDecisions)


# ── Prompt-shape generators ──────────────────────────────────────────
#
# Single source of truth: the Pydantic models above ARE the schema. The
# prompt instructions describing the JSON the LLM should produce are
# generated from those models — never hand-written and copy-pasted —
# so they can't drift.
#
# Three helpers:
#   - ``describe_action_types_markdown()`` for the Librarian's
#     ProposedAction enumeration.
#   - ``describe_librarian_response_shape()`` for the Librarian's output
#     envelope and ``describe_scholar_decisions_shape()`` for the Scholar's.


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
    AbstractEntries,
]


def _action_name(cls: type[BaseModel]) -> str:
    """Pull the ``action`` literal value off an action class.

    Pydantic stores the default on ``model_fields["action"].default``.
    """
    field = cls.model_fields.get("action")
    return str(field.default) if field is not None else cls.__name__.lower()


def _action_payload_fields(cls: type[BaseModel]) -> List[str]:
    """Non-discriminator, non-base field names for an action class, in
    declaration order. Excludes inherited fields (``action``,
    ``reasoning``, ``salience_signal``, ``existing_entry``,
    ``cited_turn_seq``) — those are documented once globally near the
    action table, not repeated on every row."""
    skip = {"action", "reasoning", "salience_signal", "existing_entry", "cited_turn_seq"}
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


def _model_schema_block(model: type[BaseModel]) -> str:
    """Return the model's JSON Schema as a fenced ``json`` block.

    Pydantic generates this from the model's fields and their
    ``Field(description=...)`` metadata. LLMs are trained on JSON
    Schema and follow it well — no hand-typed skeleton needed, no
    bespoke walker, no drift risk. Change the model, the prompt
    description updates automatically.
    """
    schema = model.model_json_schema()
    return "```json\n" + json.dumps(schema, indent=2) + "\n```"


def describe_librarian_response_shape() -> str:
    """Canonical JSON Schema for ``LibrarianProposal`` derived from the
    Pydantic model. Single source of truth: any field description on
    the model appears here automatically."""
    return _model_schema_block(LibrarianProposal)
