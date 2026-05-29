"""Scholar — gatekeeper + executor for the two-tier curator.

Architecture (typed-output era):

  - The Librarian PROPOSES a list of structural actions (write, update,
    merge, move, archive, update-readme, add-wikilink, split-folder).
  - The Scholar VERIFIES each proposal with read-only research tools and
    RETURNS the actions it approves as a typed, boundary-validated
    ``ScholarDecisions`` (vetoed proposals are simply omitted). It WRITES
    NOTHING to disk.
  - A deterministic executor (``scholar_executor.apply_decisions``) applies
    the returned actions and maintains index.md / log.md / READMEs / the
    wikilink graph.
  - After execution, deterministic post-write passes run (README
    reconciliation, wikilink repair, invariants check) to catch anything
    the Scholar missed, escalating unfixable issues back into the
    Librarian→Scholar pipeline.

The model is driven through ``typed_agent.run_typed`` over
``claude_agent_sdk`` (the subscription backend — never the metered API).
The shim hands the model the role's read-only research tools plus a
``submit_result`` tool whose schema is ``ScholarDecisions``; the
per-action Pydantic validators and the ``validate_decisions`` whole-batch
validator below reject malformed output, bouncing it back to the model via
:class:`typed_agent.ModelRetry` until it self-corrects or the retry budget
is exhausted (then the batch is dropped, exactly like a failed run).

The Scholar still owns the nudge pipeline (approved interrupts are
appended to ``~/.agent-mem/pending-nudges.md``) — that path is
orthogonal and unchanged.

``review(packets)`` keeps the signature the scheduler depends on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Set, Tuple

from . import (
    _agent_research,
    _validation,
    decay,
    priming,
    repair_queue,
    runs,
    scholar_executor,
    scholar_prompt,
)
from .llm import LLMTimeout, recursion_guard_env
from .paths import ensure_home, hot_context_path, knowledge_dir
from .typed_agent import ModelRetry, TypedAgentError, run_typed

if TYPE_CHECKING:
    from ._schemas import ScholarAction, ScholarDecisions

log = logging.getLogger("agent_mem_daemon.scholar")


# The Scholar is the precision gatekeeper — Opus tier.
SCHOLAR_MODEL = "claude-opus-4-7"

# Wall-clock budget for one Scholar agent run (matches the old SDK timeout).
SCHOLAR_TIMEOUT_S = 600.0

# Output-validation retry budget. Each ModelRetry from a validator (or a
# per-action Pydantic ValidationError) consumes one; after this many the run
# raises ``TypedAgentError`` and the daemon drops the batch (lessons recur).
OUTPUT_RETRIES = 4

# Concise framing for the model; the heavy role instructions live in the
# user prompt built by ``scholar_prompt.build_prompt`` (single source of
# truth). The shim wires ``submit_result`` whose schema is ScholarDecisions.
_SYSTEM_PROMPT = (
    "You are the Scholar, the precision gatekeeper of a two-tier memory curator. "
    "Use the read-only research tools (read_entry, grep_library, bm25_search, "
    "embedding_search) to verify the Librarian's proposals against the real "
    "library — never trust its summary. You write nothing to disk: when you have "
    "decided which proposals to approve, call submit_result EXACTLY ONCE with a "
    "ScholarDecisions object holding the approved actions. A deterministic "
    "executor applies them. Follow the detailed instructions in the user message."
)


_HETEROGENEOUS_SESSION_ID = "batch"


def _batch_session_id(packets: Sequence[Mapping[str, Any]]) -> str:
    sids = {p.get("session_id", "") for p in packets if p.get("session_id")}
    if not sids:
        return _HETEROGENEOUS_SESSION_ID
    if len(sids) == 1:
        return next(iter(sids))
    return _HETEROGENEOUS_SESSION_ID


def _all_empty(packets: Sequence[Mapping[str, Any]]) -> bool:
    """True when every packet has zero proposals AND zero interrupts."""
    for p in packets:
        if p.get("proposals") or p.get("interrupts"):
            return False
    return True


def _count_inputs(
    packets: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    """Returns (n_packets, n_proposals, n_interrupts) for logging."""
    n_props = sum(len(p.get("proposals") or []) for p in packets)
    n_ints = sum(len(p.get("interrupts") or []) for p in packets)
    return len(packets), n_props, n_ints


# ── Deps + boundary validation (whole-batch checks) ──────────────────────────


@dataclass
class ScholarDeps:
    """Dependencies the Scholar's output validator reads — the pinned
    knowledge-store root."""

    knowledge_dir: Path


def validate_decisions(deps: ScholarDeps, output: "ScholarDecisions") -> "ScholarDecisions":
    """Whole-batch checks the per-action Pydantic validators can't do in
    isolation. Raises :class:`ModelRetry` with a specific message on the first
    failure so the model fixes it and re-emits. Ported verbatim from the
    reverted Pydantic-AI migration's ``@output_validator`` (``ctx.deps`` →
    ``deps``)."""
    root = deps.knowledge_dir.resolve()
    _validate_wikilinks(output, root)
    _validate_flat_dir_caps(output, root)
    return output


def _action_body_and_path(action: "ScholarAction") -> Tuple[str, str]:
    """Return ``(body, path)`` for the body-carrying actions, or
    ``("", "")`` for the rest."""
    kind = str(action.action)
    if kind == "write_entry":
        return getattr(action, "body", ""), getattr(action, "path", "")
    if kind == "update_entry":
        return getattr(action, "new_body", ""), getattr(action, "path", "")
    if kind == "merge_entries":
        return getattr(action, "target_body", ""), getattr(action, "target_path", "")
    return "", ""


def _strip_md(path: str) -> str:
    return path[:-3] if path.endswith(".md") else path


def _created_paths(output: "ScholarDecisions") -> Set[str]:
    """The set of entry wikilink targets (no ``.md``) that this batch will
    create or relocate-to — so a body may legitimately link to a sibling
    action's output."""
    created: Set[str] = set()
    for action in output.actions:
        kind = str(action.action)
        if kind in ("write_entry", "update_entry"):
            created.add(_strip_md(getattr(action, "path", "")))
        elif kind == "merge_entries":
            created.add(_strip_md(getattr(action, "target_path", "")))
        elif kind == "move_entry":
            created.add(_strip_md(getattr(action, "to_path", "")))
    created.discard("")
    return created


def _validate_wikilinks(output: "ScholarDecisions", root: Path) -> None:
    """Every ``[[wikilink]]`` in a returned body must resolve to an existing
    entry OR to a path another action in the same batch creates. Raises
    :class:`ModelRetry` naming the first offending link."""
    created = _created_paths(output)
    for action in output.actions:
        body, path = _action_body_and_path(action)
        if not body:
            continue
        parent = (root / path).parent
        for link in _validation.body_wikilinks(body):
            if _strip_md(link) in created:
                continue
            if _validation.wikilink_resolves(link, parent, root):
                continue
            raise ModelRetry(
                f"action {action.action} for {path!r} contains an unresolvable "
                f"wikilink [[{link}]] — it points at neither an existing entry "
                f"nor a path created by another action in this batch. Fix the "
                f"target (full path from the knowledge root, no .md), add the "
                f"action that creates it, or remove the link, then re-emit."
            )


def _current_dir_counts(root: Path) -> Dict[str, int]:
    """Count entry .md files (excluding README/index/log and _archive) per
    directory, keyed by the directory's knowledge-relative posix path."""
    counts: Dict[str, int] = {}
    if not root.exists():
        return counts
    for md in root.rglob("*.md"):
        if "_archive" in md.parts:
            continue
        if md.name in ("README.md", "index.md", "log.md"):
            continue
        rel_dir = md.parent.relative_to(root).as_posix()
        counts[rel_dir] = counts.get(rel_dir, 0) + 1
    return counts


def _dir_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _apply_count_deltas(counts: Dict[str, int], output: "ScholarDecisions") -> None:
    """Mutate ``counts`` with the per-directory deltas this batch would cause
    (new writes add to their dir; moves shift between dirs; merges add the
    target and drop archived sources)."""
    for action in output.actions:
        kind = str(action.action)
        if kind == "write_entry":
            counts[_dir_of(getattr(action, "path", ""))] = (
                counts.get(_dir_of(getattr(action, "path", "")), 0) + 1
            )
        elif kind == "move_entry":
            counts[_dir_of(getattr(action, "to_path", ""))] = (
                counts.get(_dir_of(getattr(action, "to_path", "")), 0) + 1
            )
            src_dir = _dir_of(getattr(action, "from_path", ""))
            counts[src_dir] = max(0, counts.get(src_dir, 0) - 1)
        elif kind == "merge_entries":
            counts[_dir_of(getattr(action, "target_path", ""))] = (
                counts.get(_dir_of(getattr(action, "target_path", "")), 0) + 1
            )
            for src in getattr(action, "source_paths", []):
                if src == getattr(action, "target_path", ""):
                    continue
                d = _dir_of(src)
                counts[d] = max(0, counts.get(d, 0) - 1)
        elif kind == "archive_entry":
            d = _dir_of(getattr(action, "path", ""))
            counts[d] = max(0, counts.get(d, 0) - 1)


def _validate_flat_dir_caps(output: "ScholarDecisions", root: Path) -> None:
    """No directory may exceed ``MAX_FLAT_DIR_ENTRIES`` after this batch is
    applied. Raises :class:`ModelRetry` naming the first over-cap directory."""
    counts = _current_dir_counts(root)
    _apply_count_deltas(counts, output)
    for rel_dir, n in sorted(counts.items()):
        if n > _validation.MAX_FLAT_DIR_ENTRIES:
            raise ModelRetry(
                f"applying these actions would leave {n} entries in "
                f"{rel_dir}/ (cap is {_validation.MAX_FLAT_DIR_ENTRIES}). Add "
                f"move_entry action(s) to rebalance into a subfolder, or drop "
                f"the write that pushes it over, then re-emit."
            )


# ── Side-effect passes (unchanged safety net) ────────────────────────────────


def _refresh_priming_safe(packets: Sequence[Mapping[str, Any]], log_label: str) -> None:
    """Refresh ``hot-context.md`` from the rolling buffer. Swallows + logs
    any error so it can't break the rest of the review pipeline."""
    try:
        priming.refresh_hot_context(
            knowledge_dir(),
            rolling_buffer_text=priming.extract_buffer_text(packets),
            out_path=hot_context_path(),
        )
    except Exception:
        log.exception("scholar.review: priming refresh raised (%s)", log_label)


def _maybe_run_decay_sweep_safe(log_label: str) -> None:
    """Opportunistic decay sweep — runs at most once per 24h.

    Calling this on every Scholar batch is fine; ``decay.maybe_run_sweep``
    self-skips unless the cooldown has elapsed and no other sweep is in
    flight. Swallows + logs any error: the sweep is a bookkeeping
    side-effect, not part of the curation contract, so failures must
    never break the review pipeline.
    """
    try:
        result = decay.maybe_run_sweep(knowledge_dir())
    except Exception:
        log.exception("scholar.review: decay sweep raised (%s)", log_label)
        return
    if result is not None and (result.archived or result.errored):
        log.info(
            "scholar.review: decay sweep done (archived=%d kept=%d errored=%d)",
            result.archived,
            result.kept,
            result.errored,
        )


def _bump_reinforcement_counters(
    packets: Sequence[Mapping[str, Any]],
    record: runs.InvocationRecord,
) -> None:
    """Empirical reinforcement bump. Run BEFORE the model call — it's a
    fact (the user mentioned this again), not a judgment call, so we
    don't need to wait for the Scholar's deliberation."""
    try:
        reinforced_changes = scholar_prompt.apply_reinforcement_counters(
            packets,
            knowledge_dir(),
        )
        if reinforced_changes:
            record.decisions["reinforcement_bumps"] = len(reinforced_changes)
            for c in reinforced_changes:
                log.info("scholar.review: %s", c)
    except Exception:
        log.exception("scholar.review: reinforcement-counter pass raised")


def run_scholar_agent(
    prompt: str,
    knowledge_dir: Path,
    *,
    timeout_s: float = SCHOLAR_TIMEOUT_S,
) -> "tuple[ScholarDecisions, float]":
    """Run the typed Scholar agent over the SDK and return ``(decisions,
    cost_usd)``.

    The model emits a typed ``ScholarDecisions`` via the shim's
    ``submit_result`` tool; the per-action validators and ``validate_decisions``
    reject malformed output (re-prompting on any :class:`ModelRetry` up to
    ``OUTPUT_RETRIES``) before this returns. Raises :class:`LLMTimeout` if the
    wall-clock budget is exceeded, :class:`TypedAgentError` if no valid result
    is produced in budget, and propagates any other agent/model error. This is
    the seam the daemon (and the tests) drive."""
    from ._schemas import ScholarDecisions  # noqa: PLC0415 — runtime output type

    mcp_servers, allowed_tools = _agent_research.research_server_and_tools(knowledge_dir)
    deps = ScholarDeps(knowledge_dir=knowledge_dir.resolve())

    async def _run() -> "tuple[ScholarDecisions, float]":
        try:
            res = await asyncio.wait_for(
                run_typed(
                    prompt,
                    ScholarDecisions,
                    deps=deps,
                    system_prompt=_SYSTEM_PROMPT,
                    model=SCHOLAR_MODEL,
                    mcp_servers=mcp_servers,
                    allowed_tools=allowed_tools,
                    validators=[validate_decisions],
                    max_retries=OUTPUT_RETRIES,
                    cwd=knowledge_dir,
                    env=recursion_guard_env(),
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise LLMTimeout(f"Scholar agent exceeded {timeout_s}s") from e
        return res.output, res.cost_usd

    return asyncio.run(_run())


def _invoke_scholar_agent(
    prompt: str,
    knowledge_root: Path,
    record: runs.InvocationRecord,
) -> "tuple[ScholarDecisions, float] | None":
    """Run :func:`run_scholar_agent` and adapt its outcome for the review
    pipeline. Returns ``(decisions, cost_usd)`` on success or ``None`` if the
    run timed out / raised / never produced a valid result (record is marked
    errored in that case — the batch is dropped exactly like a failed run)."""
    try:
        return run_scholar_agent(prompt, knowledge_root, timeout_s=SCHOLAR_TIMEOUT_S)
    except TypedAgentError as e:
        # No valid result within the retry budget (or the model never
        # submitted): drop the batch exactly as a failed run.
        log.warning("scholar.review: agent produced no valid result (%s); batch dropped", e)
        record.mark_error(e)
        record.output_raw = ""
        return None
    except LLMTimeout as e:
        log.warning("scholar.review: agent timeout (%s); batch dropped", e)
        record.mark_error(e)
        record.output_raw = ""
        return None
    except Exception as e:
        log.exception("scholar.review: agent raised; batch dropped")
        record.mark_error(e)
        record.output_raw = ""
        return None


def _apply_decisions(
    decisions: "ScholarDecisions",
    session_id: str,
    record: runs.InvocationRecord,
) -> None:
    """Apply the validated decisions to disk via the deterministic executor,
    roll the resulting counters into the run record, and append any approved
    nudges to the pending-nudges file."""
    try:
        exec_result = scholar_executor.apply_decisions(
            decisions,
            knowledge_dir(),
            session_id=session_id,
        )
        for k, v in exec_result.counts.items():
            record.decisions[k] = record.decisions.get(k, 0) + v
        for note in exec_result.notes:
            log.info("scholar.review: executor: %s", note)
    except Exception:
        log.exception("scholar.review: executor raised")

    for k, v in scholar_prompt.summarise_decisions(decisions).items():
        # Interrupt counters (nudge / interrupt-veto) come from here; the
        # executor owns the per-action counts, so don't double-count those.
        if k in ("nudge", "interrupt-veto"):
            record.decisions[k] = record.decisions.get(k, 0) + v

    try:
        written = scholar_prompt.append_nudges_from_response(decisions)
        if written:
            record.decisions["nudges_written"] = record.decisions.get("nudges_written", 0) + len(
                written
            )
            log.info(
                "scholar.review: appended %d nudge(s) to pending-nudges.md (ids=%s)",
                len(written),
                [w["id"] for w in written],
            )
    except Exception:
        log.exception("scholar.review: nudge-file append failed")


def _reconcile_readmes_safe(record: runs.InvocationRecord) -> None:
    """Deterministic README reconciliation. Walk every folder under
    knowledge/ and ensure README.md exists with an auto-managed child
    listing that matches the actual contents. Idempotent — safe to run
    after every batch."""
    try:
        reconciled = scholar_prompt.reconcile_readmes(knowledge_dir())
        if reconciled:
            record.decisions["readmes_reconciled"] = len(reconciled)
            for c in reconciled:
                log.info("scholar.review: %s", c)
    except Exception:
        log.exception("scholar.review: README reconciliation raised")


def _escalate_unresolved_wikilink(rel_file: str, target: str, context: str) -> bool:
    """Escalation callback for a wikilink the deterministic pass cannot
    resolve. Records an integrity-repair task on the process-global queue
    so the next Librarian run researches the intended target and proposes
    the right EXISTING fix (rewrite / create-target / remove).

    Returns ``True`` when this issue is now owned by the escalation path —
    whether we just enqueued it OR an attempt for the same fingerprint is
    already in flight. Either way the caller must LEAVE the link broken so
    re-detection keeps working; neutralising would erase the signal. The
    in-flight guard inside ``RepairQueue.enqueue`` is what prevents two
    concurrent attempts for the same issue.
    """
    task = repair_queue.RepairTask(
        kind=repair_queue.KIND_BROKEN_WIKILINK,
        file=rel_file,
        target=target,
        context=context,
    )
    enqueued = repair_queue.get_queue().enqueue(task)
    if enqueued:
        log.info(
            "scholar.review: escalated unresolvable wikilink to Librarian (file=%s target=%s)",
            rel_file,
            target,
        )
    else:
        log.debug(
            "scholar.review: wikilink already in-flight; skipping duplicate "
            "escalation (file=%s target=%s)",
            rel_file,
            target,
        )
    # Owned by escalation in both cases — leave the link broken on disk.
    return True


def _repair_wikilinks_safe(record: runs.InvocationRecord) -> None:
    """Deterministic broken-wikilink repair. Removes phantom index.md rows
    and resolves broken body links the Scholar left behind. A broken link
    that cannot be deterministically resolved is ESCALATED into the
    Librarian→Scholar pipeline (via ``_escalate_unresolved_wikilink``)
    rather than silently neutralised, so Opus gets a chance to fix it
    properly; the link is left in place so it stays detectable until fixed.

    Runs BEFORE the invariants check so the safety net can confirm the
    repair actually drove broken-wikilink violations down. Integrity-first
    and idempotent (see scholar_prompt.repair_broken_wikilinks); swallows
    + logs any error so it can never break the review pipeline."""
    try:
        repaired = scholar_prompt.repair_broken_wikilinks(
            knowledge_dir(),
            on_unresolved=_escalate_unresolved_wikilink,
        )
        if repaired:
            record.decisions["wikilinks_repaired"] = len(repaired)
            for c in repaired:
                log.info("scholar.review: wikilink repair: %s", c)
    except Exception:
        log.exception("scholar.review: wikilink repair raised")


def _escalate_invariant_violation(violation: scholar_prompt.InvariantViolation) -> None:
    """Enqueue a repair task for one escalating invariant violation.

    The SAME mechanism as ``_escalate_unresolved_wikilink``: build the
    violation's :class:`repair_queue.RepairTask` and ``enqueue`` it. The
    in-flight guard inside the queue collapses repeated detections of the
    same fingerprint onto one attempt; there is no max-attempts cap, so a
    still-unfixed violation re-escalates on the next pass once its marker is
    released. Display-only violations (``repair_kind is None``) yield no
    task and are skipped."""
    task = violation.to_repair_task()
    if task is None:
        return
    enqueued = repair_queue.get_queue().enqueue(task)
    if enqueued:
        log.info(
            "scholar.review: escalated %s to Librarian (file=%s target=%s)",
            task.kind,
            task.file,
            task.target,
        )
    else:
        log.debug(
            "scholar.review: %s already in-flight; skipping duplicate escalation "
            "(file=%s target=%s)",
            task.kind,
            task.file,
            task.target,
        )


def _check_invariants_safe(record: runs.InvocationRecord) -> None:
    """Deterministic post-write invariants check + escalation. Safety net
    for anything that survived the Scholar's judgement, the reconciler, and
    the wikilink repair pass.

    Over-cap directories and bad/unparseable frontmatter are not
    deterministically fixable, so each such violation is ESCALATED into the
    Librarian→Scholar pipeline via the same repair queue + in-flight guard
    that broken wikilinks use (``_escalate_invariant_violation``). The
    violation is left in place on disk, so the next pass re-detects and
    re-escalates it until the Scholar actually fixes it."""
    try:
        violations = scholar_prompt.check_invariants_detailed(knowledge_dir())
        if violations:
            record.decisions["invariant_violations"] = len(violations)
            for v in violations:
                log.warning("scholar.review: post-write violation: %s", v.message)
                _escalate_invariant_violation(v)
        else:
            log.debug("scholar.review: invariants clean")
    except Exception:
        log.exception("scholar.review: invariants check raised")


def _collect_repair_fingerprints(
    packets: Sequence[Mapping[str, Any]],
) -> List[repair_queue.Fingerprint]:
    """Pull the ``repair_fingerprints`` an escalation attached to each
    packet at Librarian time, back into a flat list of tuples.

    Collected BEFORE any risky review work so the in-flight markers can be
    released in a ``finally`` no matter how the review exits — the attempt
    these packets represent is concluding regardless of outcome (proposal
    executed, vetoed, parse failed, or SDK error)."""
    out: List[repair_queue.Fingerprint] = []
    for p in packets:
        out.extend(repair_queue.parse_fingerprints(p.get("repair_fingerprints")))
    return out


def review(packets: Sequence[Mapping[str, Any]]) -> None:
    """Judge a batch of Librarian packets.

    Args:
        packets: a batch of Librarian-produced packets accumulated since
            the last Scholar run. May contain packets from multiple
            sessions.

    Fire-and-forget — no return value, all errors logged, never raises.
    """
    if not packets:
        log.debug("scholar.review: empty packet list; nothing to do")
        return

    # Integrity-repair escalations carried by these packets. Concluding
    # this batch RELEASES their in-flight markers (the attempt is over),
    # so collect them up front and clear in a finally that covers every
    # exit path — including the all-empty early return below and any
    # mid-review exception. The clear runs AFTER ``_repair_wikilinks_safe``
    # (whose re-detection of a still-broken link is suppressed while the
    # marker is still in-flight), so a single review never double-escalates
    # the same issue; the NEXT pass re-escalates once the marker is gone.
    repair_fps = _collect_repair_fingerprints(packets)
    try:
        _review_inner(packets)
    finally:
        if repair_fps:
            repair_queue.get_queue().clear(repair_fps)
            log.debug(
                "scholar.review: released %d in-flight repair marker(s)",
                len(repair_fps),
            )


def _review_inner(packets: Sequence[Mapping[str, Any]]) -> None:
    """Body of :func:`review` minus the repair-marker release. Split out so
    the release in ``review``'s ``finally`` covers the early-return paths
    here without duplicating the clear at each ``return``."""
    n_packets, n_props, n_ints = _count_inputs(packets)

    if _all_empty(packets):
        log.debug(
            "scholar.review: all %d packets empty (no proposals, no interrupts); "
            "skipping agent call but still refreshing priming",
            n_packets,
        )
        # Tier 1: still refresh hot-context from the buffer text even
        # when the Librarian had nothing to propose. The agent's session
        # content is signal we want to prime against regardless.
        _refresh_priming_safe(packets, "empty-packet path")
        _maybe_run_decay_sweep_safe("empty-packet path")
        return

    session_id = _batch_session_id(packets)
    ensure_home()  # create ~/.agent-mem/ if missing (side effect only)
    knowledge_root = knowledge_dir()
    prompt = scholar_prompt.build_prompt(packets)

    record = runs.InvocationRecord(
        role="scholar",
        session_id=session_id,
        input_prompt=prompt,
        input_buffer_turns=n_props + n_ints,
    )
    record.decisions = {
        "packets_in": n_packets,
        "proposals_in": n_props,
        "interrupts_in": n_ints,
    }
    started = time.time()

    _bump_reinforcement_counters(packets, record)

    log.info(
        "scholar.review: invoking agent (session=%s packets=%d proposals=%d interrupts=%d)",
        session_id,
        n_packets,
        n_props,
        n_ints,
    )

    try:
        result = _invoke_scholar_agent(prompt, knowledge_root, record)
        if result is None:
            return
        decisions, cost_usd = result

        # The agent returned a typed, boundary-validated ScholarDecisions —
        # there is no fragile JSON-scrape step. ``parsed_ok`` now means "the
        # agent produced a validated decisions object", which by construction
        # it did if we got here.
        record.parsed_ok = True
        record.cost_usd = float(cost_usd or 0.0)
        record.output_raw = decisions.model_dump_json(indent=2)

        _apply_decisions(decisions, session_id, record)

        _reconcile_readmes_safe(record)
        _repair_wikilinks_safe(record)
        _refresh_priming_safe(packets, "main path")
        _check_invariants_safe(record)
        _maybe_run_decay_sweep_safe("main path")

        duration_ms = int((time.time() - started) * 1000)
        log.info(
            "scholar.review: done (session=%s duration_ms=%d cost_usd=%.4f "
            "parsed_ok=%s decisions=%s)",
            session_id,
            duration_ms,
            record.cost_usd,
            record.parsed_ok,
            record.decisions,
        )
    finally:
        record.finalise()
