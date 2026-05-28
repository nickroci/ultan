"""The Librarian as a Pydantic AI agent.

This replaces the old Claude-Agent-SDK ``run_librarian_call`` +
hand-scraped-JSON path. The Librarian is now a typed agent that:

  - runs on ``anthropic:claude-sonnet-4-6`` (the recall tier — Haiku
    under-extracted even textbook user preferences in live testing),
  - returns a typed, validated ``LibrarianProposal`` (``output_type``): an
    ordered list of proposed actions (each with its reasoning / salience /
    evidence fields) plus the orthogonal interrupt-nudge candidates,
  - has the SAME read-only research tools as the Scholar (read an entry,
    grep the library, BM25 + embedding search), wired as direct in-process
    ``@agent.tool``s via ``_agent_common`` — it has NO file-writing tools;
    it only PROPOSES, the Scholar approves/vetoes and the executor writes,
  - validates its own output at the boundary via an ``@agent.output_validator``
    that bounces back MALFORMED proposals — a path that escapes the
    knowledge root or isn't a ``.md`` entry, or a body whose YAML
    frontmatter does not parse — via ``ModelRetry`` with a specific message.

The boundary bar here is deliberately LOWER than the Scholar's: the
Librarian is a generous recall layer (Sonnet-tier) and the Scholar is the
precision gatekeeper (Opus-tier). We only reject proposals that are
objectively un-executable regardless of judgement — well-formedness, not
salience or completeness. Completeness (every required frontmatter field,
id↔slug agreement, wikilink resolution) is enforced on the SCHOLAR's
output, not here, so a maybe-good proposal is never burned on a retry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from pydantic_ai import Agent, ModelRetry, RunContext

from . import _agent_common, _validation
from ._agent_common import ResearchDeps as LibrarianDeps  # the Librarian's deps type
from ._schemas import (
    LibrarianProposal,
    MergeEntries,
    ProposedActionT,
    UpdateEntry,
    WriteEntry,
)

log = logging.getLogger("agent_mem_daemon.librarian_agent")

LIBRARIAN_MODEL = "anthropic:claude-sonnet-4-6"

# Wall-clock budget for one Librarian agent run (matches the old SDK timeout).
LIBRARIAN_TIMEOUT_S = 600.0

# Output-validation retry budget. Each ModelRetry from the boundary validator
# (or a per-action Pydantic ValidationError) consumes one; after this many the
# run raises and the daemon emits an empty packet (signal recurs next session).
OUTPUT_RETRIES = 3


# The Librarian agent, built once at module import. ``defer_model_check=True``
# lets the daemon import this module without an ``ANTHROPIC_API_KEY`` present.
# The read-only research tools are shared with the Scholar (registered via
# ``_agent_common``); the output validator below is Librarian-specific.
LIBRARIAN_AGENT: "Agent[LibrarianDeps, LibrarianProposal]" = Agent(
    LIBRARIAN_MODEL,
    output_type=LibrarianProposal,
    deps_type=LibrarianDeps,
    retries={"output": OUTPUT_RETRIES},
    defer_model_check=True,
)

_agent_common.register_research_tools(LIBRARIAN_AGENT)


# ── Boundary validation (well-formed paths + parseable bodies) ───────────


@LIBRARIAN_AGENT.output_validator
def validate_proposal(
    ctx: RunContext[LibrarianDeps], output: LibrarianProposal
) -> LibrarianProposal:
    """Reject proposals the executor could never apply, regardless of the
    Scholar's later judgement. Raises ``ModelRetry`` with a specific,
    actionable message on the first offending proposal so the Librarian
    self-corrects and re-emits.

    Two boundary checks only (recall over precision — the Scholar owns the
    rest): every proposal's target path(s) must be well-formed
    (knowledge-relative, no escape, ``.md`` for entry actions), and any body
    a body-carrying proposal supplies must have YAML frontmatter that
    parses."""
    root = ctx.deps.knowledge_dir.resolve()
    for proposal in output.proposals:
        _validate_proposal_paths(proposal, root)
        _validate_proposal_body(proposal)
    return output


# ── Output-validator helpers (module-level, pure) ────────────────────────


def _entry_target_paths(proposal: "ProposedActionT") -> List[Tuple[str, str]]:
    """Return ``(field_name, path)`` pairs for every entry-file path a
    proposal carries that must be a well-formed ``.md`` path. Folder-only
    paths (``update_readme``/``split_folder`` folders) and the loose
    ``existing_entry`` citation are excluded — those are not entry targets
    the executor writes a file to from this value."""
    kind = str(proposal.action)
    pairs: List[Tuple[str, str]] = []
    if kind == "write_entry":
        pairs.append(("path", getattr(proposal, "path", "")))
    elif kind == "update_entry":
        pairs.append(("path", getattr(proposal, "path", "")))
    elif kind == "merge_entries":
        pairs.append(("target_path", getattr(proposal, "target_path", "")))
        for src in getattr(proposal, "source_paths", []) or []:
            pairs.append(("source_paths", str(src)))
    elif kind == "move_entry":
        pairs.append(("from_path", getattr(proposal, "from_path", "")))
        pairs.append(("to_path", getattr(proposal, "to_path", "")))
    elif kind == "archive_entry":
        pairs.append(("path", getattr(proposal, "path", "")))
    elif kind == "deprecate_entry":
        pairs.append(("path", getattr(proposal, "path", "")))
        pairs.append(("superseded_by", getattr(proposal, "superseded_by", "")))
    elif kind == "add_wikilink":
        pairs.append(("from_path", getattr(proposal, "from_path", "")))
        pairs.append(("to_path", getattr(proposal, "to_path", "")))
    return [(name, path) for name, path in pairs if path]


def _path_wellformed_error(path: str, root: Path) -> str | None:
    """Return a one-line reason ``path`` is not a well-formed knowledge-
    relative ``.md`` entry path, or ``None`` when it is fine. Rejects
    absolute paths, parent-escape (``..``), and non-``.md`` targets."""
    if path.startswith("/") or path.startswith("~"):
        return f"path {path!r} must be RELATIVE to the knowledge root, not absolute"
    candidate = (root / path).resolve()
    if not _agent_common.inside(root, candidate):
        return f"path {path!r} resolves OUTSIDE the knowledge store (no '..' escapes)"
    if not path.endswith(".md"):
        return f"entry path {path!r} must end in '.md'"
    return None


def _validate_proposal_paths(proposal: "ProposedActionT", root: Path) -> None:
    """Raise ``ModelRetry`` on the first malformed entry path in a proposal."""
    for field_name, path in _entry_target_paths(proposal):
        err = _path_wellformed_error(path, root)
        if err is not None:
            raise ModelRetry(
                f"proposal {proposal.action} has a malformed {field_name}: {err}. "
                f"All paths are relative to the knowledge root (e.g. "
                f"'global/python/use-uv.md'); fix it and re-emit."
            )


def _validate_proposal_body(proposal: "ProposedActionT") -> None:
    """For a body-carrying proposal, raise ``ModelRetry`` if a supplied body
    has no parseable YAML frontmatter. An empty body is left to the Scholar
    (the Librarian may legitimately propose a thin write the Scholar fleshes
    out); only a NON-empty body that fails to parse is a boundary defect."""
    if isinstance(proposal, (WriteEntry, UpdateEntry)):
        body = proposal.body if isinstance(proposal, WriteEntry) else proposal.new_body
        path = proposal.path
    elif isinstance(proposal, MergeEntries):
        body = proposal.target_body
        path = proposal.target_path
    else:
        return
    if not body.strip():
        return
    if not _validation.parse_frontmatter(body):
        raise ModelRetry(
            f"proposal {proposal.action} for {path!r} has a body with no parseable "
            f"YAML frontmatter (expected a leading '---' ... '---' block). Fix the "
            f"frontmatter so it parses, then re-emit."
        )


def run_librarian_agent(
    prompt: str,
    knowledge_dir: Path,
    *,
    timeout_s: float = LIBRARIAN_TIMEOUT_S,
) -> Tuple["LibrarianProposal", float]:
    """Run the Librarian agent on ``prompt`` and return
    ``(validated_proposal, cost_usd)``.

    The model emits a typed ``LibrarianProposal``; Pydantic AI runs the
    per-action validators and the ``output_validator`` (re-prompting the
    model on ``ModelRetry`` up to the output retry budget) before returning.
    Raises :class:`LLMTimeout` if the wall-clock budget is exceeded, and
    propagates any other agent/model error for the caller to log.
    """
    deps = LibrarianDeps(knowledge_dir=knowledge_dir.resolve())
    return _agent_common.run_agent_to_output(
        LIBRARIAN_AGENT,
        prompt,
        deps,
        model_ref=LIBRARIAN_MODEL,
        timeout_s=timeout_s,
        role="Librarian",
    )
