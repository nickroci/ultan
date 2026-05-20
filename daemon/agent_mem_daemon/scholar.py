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
from typing import List

from . import priming
from . import runs
from . import scholar_prompt
from .librarian import EvidencePacket
from .llm import LLMTimeout, SCHOLAR_TIMEOUT_S, run_scholar_call
from .paths import ensure_home, hot_context_path, knowledge_dir


log = logging.getLogger("agent_mem_daemon.scholar")


_HETEROGENEOUS_SESSION_ID = "batch"


def _batch_session_id(packets: List[EvidencePacket]) -> str:
    sids = {p.get("session_id", "") for p in packets if p.get("session_id")}
    if not sids:
        return _HETEROGENEOUS_SESSION_ID
    if len(sids) == 1:
        return next(iter(sids))
    return _HETEROGENEOUS_SESSION_ID


def _all_empty(packets: List[EvidencePacket]) -> bool:
    """True when every packet has zero proposals AND zero interrupts."""
    for p in packets:
        if p.get("proposals") or p.get("interrupts"):
            return False
    return True


def _count_inputs(packets: List[EvidencePacket]) -> tuple[int, int, int]:
    """Returns (n_packets, n_proposals, n_interrupts) for logging."""
    n_props = sum(len(p.get("proposals") or []) for p in packets)
    n_ints = sum(len(p.get("interrupts") or []) for p in packets)
    return len(packets), n_props, n_ints


def review(packets: List[EvidencePacket]) -> None:
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
        try:
            priming.refresh_hot_context(
                knowledge_dir(),
                rolling_buffer_text=priming.extract_buffer_text(packets),
                out_path=hot_context_path(),
            )
        except Exception:
            log.exception("scholar.review: priming refresh raised (empty-packet path)")
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

    # ── Empirical reinforcement counter bump ─────────────────────
    # Run BEFORE the SDK call. Reinforcement is a fact ("the user
    # mentioned this again"), not a judgment call — no need to wait
    # for the Scholar. Bumps an entry's `reinforced` counter and
    # stamps `last_reinforced` when the Librarian's proposal flagged
    # `salience_signal: "reinforces"` with a valid existing_entry path.
    try:
        reinforced_changes = scholar_prompt.apply_reinforcement_counters(
            packets, knowledge_dir(),
        )
        if reinforced_changes:
            record.decisions["reinforcement_bumps"] = len(reinforced_changes)
            for c in reinforced_changes:
                log.info("scholar.review: %s", c)
    except Exception:
        log.exception("scholar.review: reinforcement-counter pass raised")

    log.info(
        "scholar.review: invoking SDK (session=%s packets=%d proposals=%d interrupts=%d)",
        session_id, n_packets, n_props, n_ints,
    )

    try:
        try:
            response_text, cost_usd = run_scholar_call(
                prompt,
                cwd=knowledge_root,
                timeout_s=SCHOLAR_TIMEOUT_S,
            )
        except LLMTimeout as e:
            log.warning("scholar.review: SDK timeout (%s); batch dropped", e)
            record.mark_error(e)
            record.output_raw = ""
            return
        except Exception as e:
            log.exception("scholar.review: SDK raised; batch dropped")
            record.mark_error(e)
            record.output_raw = ""
            return

        record.output_raw = response_text or ""
        record.cost_usd = float(cost_usd or 0.0)

        parsed, ok = scholar_prompt.parse_response(record.output_raw)
        record.parsed_ok = ok
        if not ok:
            log.warning(
                "scholar.review: final-JSON parse failed "
                "(session=%s response_chars=%d); continuing without counters",
                session_id, len(record.output_raw),
            )
        else:
            decisions = scholar_prompt.summarise_decisions(parsed)
            for k, v in decisions.items():
                record.decisions[k] = record.decisions.get(k, 0) + v

            # Side-effect: append approved interrupts.
            try:
                written = scholar_prompt.append_nudges_from_response(parsed)
                if written:
                    record.decisions["nudges_written"] = (
                        record.decisions.get("nudges_written", 0) + len(written)
                    )
                    log.info(
                        "scholar.review: appended %d nudge(s) to pending-nudges.md "
                        "(ids=%s)",
                        len(written), [w["id"] for w in written],
                    )
            except Exception:
                log.exception("scholar.review: nudge-file append failed")

        # ── Deterministic README reconciliation ────────────────────
        # Walk every folder under knowledge/ and ensure README.md
        # exists with an auto-managed child listing that matches the
        # actual contents. This guarantees the "every folder has a
        # README" invariant and keeps parent listings in sync when
        # the Librarian forgets to propose UpdateReadme for every
        # level of the chain. Idempotent — safe to run after every
        # batch.
        try:
            reconciled = scholar_prompt.reconcile_readmes(knowledge_dir())
            if reconciled:
                record.decisions["readmes_reconciled"] = len(reconciled)
                for c in reconciled:
                    log.info("scholar.review: %s", c)
        except Exception:
            log.exception("scholar.review: README reconciliation raised")

        # ── Tier 1: ambient priming refresh ────────────────────────
        # After the library state has settled (proposals executed,
        # READMEs reconciled), re-pick the top-K most relevant entries
        # for what THIS batch was about and write them to the hot-
        # context file. The UserPromptSubmit hook reads that file on
        # the next turn and injects it as additionalContext — gives
        # the agent passive "familiarity" with the most-likely-
        # relevant library entries without any extra LLM cost.
        try:
            priming.refresh_hot_context(
                knowledge_dir(),
                rolling_buffer_text=priming.extract_buffer_text(packets),
                out_path=hot_context_path(),
            )
        except Exception:
            log.exception("scholar.review: priming refresh raised")

        # ── Deterministic post-write invariants check ──────────────
        # Safety net: log a WARNING for any violation that survived
        # both the Scholar's own judgement and the reconciler. By the
        # time we reach here it should be very rare.
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

        duration_ms = int((time.time() - started) * 1000)
        log.info(
            "scholar.review: done (session=%s duration_ms=%d cost_usd=%.4f "
            "parsed_ok=%s decisions=%s)",
            session_id, duration_ms, record.cost_usd,
            record.parsed_ok, record.decisions,
        )
    finally:
        record.finalise()
