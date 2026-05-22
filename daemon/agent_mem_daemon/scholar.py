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
from typing import Any, Mapping, Sequence

from . import priming, runs, scholar_prompt
from .llm import SCHOLAR_TIMEOUT_S, LLMTimeout, run_scholar_call
from .paths import ensure_home, hot_context_path, knowledge_dir

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


def _invoke_scholar_sdk(
    prompt: str,
    knowledge_root: Path,
    record: runs.InvocationRecord,
) -> tuple[str, float] | None:
    """Run the SDK call. Returns ``(response_text, cost_usd)`` on success
    or ``None`` if the call timed out / raised (record is marked errored
    in that case)."""
    try:
        return run_scholar_call(
            prompt,
            cwd=knowledge_root,
            timeout_s=SCHOLAR_TIMEOUT_S,
        )
    except LLMTimeout as e:
        log.warning("scholar.review: SDK timeout (%s); batch dropped", e)
        record.mark_error(e)
        record.output_raw = ""
        return None
    except Exception as e:
        log.exception("scholar.review: SDK raised; batch dropped")
        record.mark_error(e)
        record.output_raw = ""
        return None


def _apply_parsed_response(
    parsed: Any,
    record: runs.InvocationRecord,
) -> None:
    """Roll the Scholar's parsed JSON into the run record and append any
    approved nudges to the pending-nudges file."""
    decisions = scholar_prompt.summarise_decisions(parsed)
    for k, v in decisions.items():
        record.decisions[k] = record.decisions.get(k, 0) + v

    try:
        written = scholar_prompt.append_nudges_from_response(parsed)
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


def _check_invariants_safe(record: runs.InvocationRecord) -> None:
    """Deterministic post-write invariants check. Safety net for anything
    that survived both the Scholar's judgement and the reconciler."""
    try:
        violations = scholar_prompt.check_invariants(knowledge_dir())
        if violations:
            record.decisions["invariant_violations"] = len(violations)
            for v in violations:
                log.warning("scholar.review: post-write violation: %s", v)
        else:
            log.debug("scholar.review: invariants clean")
    except Exception:
        log.exception("scholar.review: invariants check raised")


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
        return

    session_id = _batch_session_id(packets)
    knowledge_root = ensure_home()
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
        "scholar.review: invoking SDK (session=%s packets=%d proposals=%d interrupts=%d)",
        session_id,
        n_packets,
        n_props,
        n_ints,
    )

    try:
        result = _invoke_scholar_sdk(prompt, knowledge_root, record)
        if result is None:
            return
        response_text, cost_usd = result

        record.output_raw = response_text or ""
        record.cost_usd = float(cost_usd or 0.0)

        parsed, ok = scholar_prompt.parse_response(record.output_raw)
        record.parsed_ok = ok
        if not ok:
            log.warning(
                "scholar.review: final-JSON parse failed "
                "(session=%s response_chars=%d); continuing without counters",
                session_id,
                len(record.output_raw),
            )
        else:
            _apply_parsed_response(parsed, record)

        _reconcile_readmes_safe(record)
        _refresh_priming_safe(packets, "main path")
        _check_invariants_safe(record)

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
