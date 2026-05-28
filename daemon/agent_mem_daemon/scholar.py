"""Scholar — gatekeeper + executor for the two-tier curator.

New architecture (vs. the old "Scholar picks the action" design):

  - The Librarian PROPOSES a list of structural actions (write, update,
    merge, move, archive, update-readme, add-wikilink, split-folder).
  - The Scholar APPROVES (and executes via Write/Edit) or VETOES (drops
    with a one-sentence reason). No fix-ups — drop-only behaviour.
  - After execution, a deterministic post-write validation pass runs
    (``scholar_prompt.check_invariants``) to catch anything the Scholar
    missed.

The Scholar still owns the nudge pipeline (approved interrupts are
appended to ``~/.agent-mem/pending-nudges.md``) — that path is
orthogonal and unchanged.

``review(packets)`` keeps the signature the scheduler depends on.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence

from . import decay, priming, repair_queue, runs, scholar_executor, scholar_prompt
from .llm import LLMTimeout
from .paths import ensure_home, hot_context_path, knowledge_dir
from .scholar_agent import SCHOLAR_TIMEOUT_S, run_scholar_agent

if TYPE_CHECKING:
    from ._schemas import ScholarDecisions

log = logging.getLogger("agent_mem_daemon.scholar")


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
    """Empirical reinforcement bump. Run BEFORE the SDK call — it's a
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


def _invoke_scholar_agent(
    prompt: str,
    knowledge_root: Path,
    record: runs.InvocationRecord,
) -> "tuple[ScholarDecisions, float] | None":
    """Run the Pydantic AI Scholar agent. Returns ``(decisions, cost_usd)``
    on success or ``None`` if the run timed out / raised (record is marked
    errored in that case). The returned ``ScholarDecisions`` is already
    boundary-validated by the agent's per-action validators + output
    validator (the model was re-prompted on any ModelRetry)."""
    try:
        return run_scholar_agent(
            prompt,
            knowledge_root,
            timeout_s=SCHOLAR_TIMEOUT_S,
        )
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
            "skipping SDK call but still refreshing priming",
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
