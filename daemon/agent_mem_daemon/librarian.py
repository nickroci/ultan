"""Librarian — the active organiser (typed-output era).

Runs on every ``Stop`` event. Given a rolling-buffer snapshot, it:

1. Flattens the snapshot into ``[turn_id] [role] <text>`` lines.
2. Builds a library snapshot (tree + READMEs + index excerpt).
3. Assembles the prompt and runs the typed Librarian agent over
   ``claude_agent_sdk`` (the subscription backend — never the metered API)
   via ``typed_agent.run_typed``, with read-only research tools
   (read_entry / grep_library / bm25_search / embedding_search) so it can
   inspect specific entries before proposing actions. No regex pre-pass, no
   BM25 seed hits — the model decides what's worth checking via its tools.
4. The agent returns a typed, boundary-validated ``LibrarianProposal``;
   ``scan`` dumps it into the ``proposals`` / ``interrupts`` dict lists the
   Scholar pipeline consumes — there is no fragile JSON-scrape step.

The Librarian PROPOSES actions; the Scholar approves or vetoes each one.
The Librarian writes nothing to disk.

The boundary bar here is deliberately LOWER than the Scholar's: the
Librarian is a generous recall layer (Sonnet-tier) and the Scholar is the
precision gatekeeper (Opus-tier). The output validator only rejects
proposals that are objectively un-executable regardless of judgement —
well-formedness, not salience or completeness — so a maybe-good proposal is
never burned on a retry.

The ``EvidencePacket`` shape and the ``scan(buffer_snapshot)`` signature
are frozen contracts the scheduler depends on. The dict contents carry
``proposals`` and ``interrupts``, but the function signature does not change.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, TypedDict

from typing_extensions import NotRequired

from . import _agent_research, _validation, repair_queue, runs
from . import librarian_prompt as lp
from ._schemas import MergeEntries, ProposedActionT, UpdateEntry, WriteEntry
from .llm import LLMTimeout
from .paths import ensure_home, knowledge_dir
from .typed_agent import ModelRetry, TypedAgentError, run_typed

if TYPE_CHECKING:
    from ._schemas import LibrarianProposal

log = logging.getLogger("agent_mem_daemon.librarian")


# The Librarian is the recall tier — Sonnet (Haiku under-extracted even
# textbook user preferences in live testing).
LIBRARIAN_MODEL = "claude-sonnet-4-6"

# Wall-clock budget for one Librarian agent run (matches the old SDK timeout).
LIBRARIAN_TIMEOUT_S = 600.0

# Output-validation retry budget. Each ModelRetry from the boundary validator
# (or a per-action Pydantic ValidationError) consumes one; after this many the
# run raises ``TypedAgentError`` and the daemon emits an empty packet (signal
# recurs next session).
OUTPUT_RETRIES = 3

_SYSTEM_PROMPT = (
    "You are the Librarian, the recall tier of a two-tier memory curator. "
    "Use the read-only research tools (read_entry, grep_library, bm25_search, "
    "embedding_search) to inspect the existing library before proposing "
    "actions. You write nothing to disk and you only PROPOSE — the Scholar "
    "approves or vetoes. When ready, call submit_result EXACTLY ONCE with a "
    "LibrarianProposal object. Follow the detailed instructions in the user "
    "message."
)


class EvidencePacket(TypedDict):
    """Frozen shape the Scholar reads.

    Items inside ``proposals`` match the ``LibrarianProposal.proposals``
    schema (see ``_schemas.py``); items inside ``interrupts`` match
    ``LibrarianInterrupt``.

    ``session_id`` / ``proposals`` / ``interrupts`` are always set by
    ``_empty_packet`` — there is no valid packet missing any of them.

    ``repair_fingerprints`` is OPTIONAL: present only when this packet was
    produced by a Librarian run that drained integrity-repair tasks from
    ``repair_queue``. It carries the ``(kind, file, target)`` fingerprints
    of the issues this run took ownership of, so the Scholar can RELEASE
    their in-flight markers once it concludes the review. It is attached to
    EVERY packet the run emits — including the error/empty fallback — so a
    drained task can never leak its marker (a Librarian that drained but
    then failed must still hand the fingerprints to the Scholar to clear).
    """

    session_id: str
    proposals: List[Dict[str, Any]]
    interrupts: List[Dict[str, Any]]
    repair_fingerprints: NotRequired[List[repair_queue.Fingerprint]]


def _empty_packet(session_id: str) -> EvidencePacket:
    return EvidencePacket(
        session_id=session_id,
        proposals=[],
        interrupts=[],
    )


# ── Deps + boundary validation (well-formed paths + parseable bodies) ────────


@dataclass
class LibrarianDeps:
    """Dependencies the Librarian's output validator reads — the pinned
    knowledge-store root."""

    knowledge_dir: Path


def validate_proposal(deps: LibrarianDeps, output: "LibrarianProposal") -> "LibrarianProposal":
    """Reject proposals the executor could never apply, regardless of the
    Scholar's later judgement. Raises :class:`ModelRetry` with a specific,
    actionable message on the first offending proposal so the Librarian
    self-corrects and re-emits. Ported verbatim from the reverted Pydantic-AI
    migration's ``@output_validator`` (``ctx.deps`` → ``deps``).

    Two boundary checks only (recall over precision — the Scholar owns the
    rest): every proposal's target path(s) must be well-formed
    (knowledge-relative, no escape, ``.md`` for entry actions), and any body
    a body-carrying proposal supplies must have YAML frontmatter that
    parses."""
    root = deps.knowledge_dir.resolve()
    for proposal in output.proposals:
        _validate_proposal_paths(proposal, root)
        _validate_proposal_body(proposal)
    return output


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
    if not _agent_research.inside(root, candidate):
        return f"path {path!r} resolves OUTSIDE the knowledge store (no '..' escapes)"
    if not path.endswith(".md"):
        return f"entry path {path!r} must end in '.md'"
    return None


def _validate_proposal_paths(proposal: "ProposedActionT", root: Path) -> None:
    """Raise :class:`ModelRetry` on the first malformed entry path in a
    proposal."""
    for field_name, path in _entry_target_paths(proposal):
        err = _path_wellformed_error(path, root)
        if err is not None:
            raise ModelRetry(
                f"proposal {proposal.action} has a malformed {field_name}: {err}. "
                f"All paths are relative to the knowledge root (e.g. "
                f"'global/python/use-uv.md'); fix it and re-emit."
            )


def _validate_proposal_body(proposal: "ProposedActionT") -> None:
    """For a body-carrying proposal, raise :class:`ModelRetry` if a supplied
    body has no parseable YAML frontmatter. An empty body is left to the
    Scholar (the Librarian may legitimately propose a thin write the Scholar
    fleshes out); only a NON-empty body that fails to parse is a boundary
    defect."""
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


# ── Agent invocation ─────────────────────────────────────────────────────────


def run_librarian_agent(
    prompt: str,
    knowledge_dir: Path,
    *,
    timeout_s: float = LIBRARIAN_TIMEOUT_S,
) -> "Tuple[LibrarianProposal, float]":
    """Run the typed Librarian agent over the SDK and return
    ``(validated_proposal, cost_usd)``.

    The model emits a typed ``LibrarianProposal`` via the shim's
    ``submit_result`` tool; the per-action validators and ``validate_proposal``
    reject malformed output (re-prompting on any :class:`ModelRetry` up to
    ``OUTPUT_RETRIES``) before this returns. Raises :class:`LLMTimeout` if the
    wall-clock budget is exceeded, :class:`TypedAgentError` if no valid result
    is produced in budget, and propagates any other agent/model error for the
    caller to log. This is the seam the daemon (and the tests) drive."""
    from ._schemas import LibrarianProposal  # noqa: PLC0415 — runtime output type

    mcp_servers, allowed_tools = _agent_research.research_server_and_tools(knowledge_dir)
    deps = LibrarianDeps(knowledge_dir=knowledge_dir.resolve())

    async def _run() -> "Tuple[LibrarianProposal, float]":
        try:
            res = await asyncio.wait_for(
                run_typed(
                    prompt,
                    LibrarianProposal,
                    deps=deps,
                    system_prompt=_SYSTEM_PROMPT,
                    model=LIBRARIAN_MODEL,
                    mcp_servers=mcp_servers,
                    allowed_tools=allowed_tools,
                    validators=[validate_proposal],
                    max_retries=OUTPUT_RETRIES,
                    cwd=knowledge_dir,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise LLMTimeout(f"Librarian agent exceeded {timeout_s}s") from e
        return res.output, res.cost_usd

    return asyncio.run(_run())


def scan(buffer_snapshot: Dict[str, Any]) -> EvidencePacket:
    """Run one Librarian pass over a buffer snapshot.

    Returns an EvidencePacket. Never raises — exceptions are logged and
    turned into an empty packet. The scheduler relies on this guarantee.

    Integrity-repair escalation: before doing anything else, this drains
    any pending repair tasks from ``repair_queue`` and renders them into
    the prompt so the Librarian researches each broken target and proposes
    the right EXISTING fix. The drained fingerprints are attached to
    WHATEVER packet this run emits — success, empty, or error — so the
    Scholar can always release their in-flight markers; a drained task that
    fell on the floor would otherwise stay in-flight forever and block
    re-escalation.
    """
    session_id = str(buffer_snapshot.get("session_id") or "?")

    # Drain BEFORE the work so the fingerprints ride out on every return
    # path. Attaching them to the final packet is centralised at the bottom.
    repair_tasks = repair_queue.get_queue().drain_pending()
    fingerprints = repair_queue.fingerprints_of(repair_tasks)
    if repair_tasks:
        log.info(
            "librarian.scan: session=%s carrying %d integrity-repair task(s)",
            session_id,
            len(repair_tasks),
        )

    packet = _scan_for_packet(buffer_snapshot, session_id, repair_tasks)
    if fingerprints:
        packet["repair_fingerprints"] = fingerprints
    return packet


def _scan_for_packet(
    buffer_snapshot: Dict[str, Any],
    session_id: str,
    repair_tasks: List[repair_queue.RepairTask],
) -> EvidencePacket:
    """Run one Librarian pass and return the raw packet (no fingerprint
    attachment — :func:`scan` owns that so it happens on every path)."""
    record = runs.InvocationRecord(role="librarian", session_id=session_id)
    packet: EvidencePacket = _empty_packet(session_id)

    try:
        # ── Step 1: flatten the buffer ─────────────────────────────
        formatted_buffer, flat = lp.buffer_to_prompt_text(buffer_snapshot)
        record.input_buffer_turns = len(buffer_snapshot.get("turns") or [])

        # If the buffer has no quotable text AND there are no repair tasks,
        # short-circuit. When repair tasks are present we still invoke the
        # LLM — the escalated issue is itself the reason to run.
        if not flat and not repair_tasks:
            log.debug(
                "librarian.scan: session=%s has no quotable turns; skipping LLM call",
                session_id,
            )
            record.parsed_ok = True
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,
                "skipped_no_buffer_text": 1,
            }
            return packet

        # ── Step 2: knowledge-store inputs ─────────────────────────
        # No regex pre-pass, no BM25 seed hits. The Librarian has its
        # read-only research tools and a library snapshot — it decides what
        # to inspect for dedup itself. Smart model, not pattern matching.
        kdir = knowledge_dir()
        library_snapshot = lp.build_library_snapshot(kdir)
        applies_when_table = lp.build_applies_when_table(kdir)
        # Bucket = the on-disk directory name everyone should agree on
        # (librarian's path proposals, scholar's writes, priming's
        # scope boost, the nudge filter). Resolved through the single
        # ``aliases.session_bucket`` entry point so any consistency
        # drift between layers gets caught here.
        bucket = lp.derive_project_bucket(buffer_snapshot)

        # ── Step 3: assemble + invoke ──────────────────────────────
        prompt = lp.assemble_prompt(
            project_slug=bucket,
            rolling_buffer=formatted_buffer,
            library_snapshot=library_snapshot,
            applies_when_table=applies_when_table,
            repair_tasks=lp.format_repair_tasks(repair_tasks),
        )
        record.input_prompt = prompt

        log.debug(
            "librarian.scan: session=%s bucket=%s turns=%d repair_tasks=%d prompt_chars=%d",
            session_id,
            bucket,
            len(flat),
            len(repair_tasks),
            len(prompt),
        )

        # The Librarian needs the knowledge dir so its read/grep/bm25/
        # embedding tools land in the right tree. ensure_home() creates the
        # dir if missing so we never pass a nonexistent path.
        ensure_home()
        try:
            proposal, cost_usd = run_librarian_agent(prompt, kdir, timeout_s=LIBRARIAN_TIMEOUT_S)
        except LLMTimeout as e:
            log.warning("librarian agent timeout for session=%s: %s", session_id, e)
            record.mark_error(e)
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,
                "llm_timeout": 1,
            }
            return packet
        except TypedAgentError as e:
            log.warning(
                "librarian agent produced no valid result for session=%s: %s; "
                "emitting empty packet",
                session_id,
                e,
            )
            record.mark_error(e)
            record.parsed_ok = False
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,
                "parse_failed": 1,
            }
            return packet
        except Exception as e:
            log.exception("librarian agent call failed for session=%s", session_id)
            record.mark_error(e)
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,
                "llm_error": 1,
            }
            return packet

        record.cost_usd = float(cost_usd or 0.0)

        # ── Step 4: dump the typed proposal into the packet's dict lists ──
        # The agent already validated structure + boundaries (per-action
        # validators + the output validator, re-prompted on ModelRetry), so
        # there is no JSON-scrape step. ``parsed_ok`` now means "the agent
        # produced a validated LibrarianProposal", which by construction it
        # did if we got here.
        dumped = proposal.model_dump()
        proposals: List[Dict[str, Any]] = dumped.get("proposals") or []
        interrupts: List[Dict[str, Any]] = dumped.get("interrupts") or []
        record.output_raw = proposal.model_dump_json(indent=2)
        packet = EvidencePacket(
            session_id=session_id,
            proposals=proposals,
            interrupts=interrupts,
        )
        record.parsed_ok = True
        record.decisions = {
            "proposals": len(proposals),
            "interrupts": len(interrupts),
        }
        log.info(
            "librarian.scan: session=%s emitted %d proposal(s), %d interrupt(s)",
            session_id,
            len(proposals),
            len(interrupts),
        )
        return packet

    except Exception as e:
        log.exception("librarian.scan unexpected error; returning empty packet")
        record.mark_error(e)
        return _empty_packet(session_id)

    finally:
        record.finalise()
