"""Librarian — the active organiser (new architecture).

Runs on every ``Stop`` event. Given a rolling-buffer snapshot, it:

1. Flattens the snapshot into ``[turn_id] [role] <text>`` lines.
2. Builds a library snapshot (tree + READMEs + index excerpt).
3. Assembles the prompt and runs the Pydantic AI ``librarian_agent`` (with
   read-only research tools — read/grep/bm25/embedding — so the Librarian
   can inspect specific entries before proposing actions). No regex
   pre-pass, no BM25 seed hits — the model decides what's worth checking
   via its own tools.
4. The agent returns a typed, boundary-validated ``LibrarianProposal``;
   ``scan`` dumps it into the ``proposals`` / ``interrupts`` dict lists the
   Scholar pipeline consumes — there is no fragile JSON-scrape step.

The Librarian PROPOSES actions; the Scholar approves or vetoes each
one. The Librarian writes nothing to disk.

The ``EvidencePacket`` shape and the ``scan(buffer_snapshot)`` signature
are frozen contracts the scheduler depends on. The dict contents
change to carry ``proposals`` (new) and ``interrupts`` (unchanged), but
the function signature does not.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from typing_extensions import NotRequired

from . import librarian_agent, repair_queue, runs
from . import librarian_prompt as lp
from .llm import LLMTimeout
from .paths import ensure_home, knowledge_dir

log = logging.getLogger("agent_mem_daemon.librarian")


_LIBRARIAN_TIMEOUT_S = librarian_agent.LIBRARIAN_TIMEOUT_S


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
            proposal, cost_usd = librarian_agent.run_librarian_agent(
                prompt, kdir, timeout_s=_LIBRARIAN_TIMEOUT_S
            )
        except LLMTimeout as e:
            log.warning("librarian agent timeout for session=%s: %s", session_id, e)
            record.mark_error(e)
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,
                "llm_timeout": 1,
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
