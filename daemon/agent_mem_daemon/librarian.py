"""Librarian — the active organiser (new architecture).

Runs on every ``Stop`` event. Given a rolling-buffer snapshot, it:

1. Flattens the snapshot into ``[turn_id] [role] <text>`` lines.
2. Builds a library snapshot (tree + READMEs + index excerpt).
3. Assembles the prompt and calls ``llm.run_librarian_call`` (with
   Read+Glob tools enabled so the Librarian can inspect specific
   entries before proposing actions). No regex pre-pass, no BM25 seed
   hits — the model decides what's worth checking via its own tools.
4. Parses the response into a `LibrarianProposal` packet the Scholar
   can consume.

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

from . import librarian_prompt as lp
from . import llm
from . import runs
from .paths import knowledge_dir, ensure_home


log = logging.getLogger("agent_mem_daemon.librarian")


_LIBRARIAN_TIMEOUT_S = llm.LIBRARIAN_TIMEOUT_S


class EvidencePacket(TypedDict, total=False):
    """Frozen shape the Scholar reads.

    Items inside ``proposals`` match the ``LibrarianProposal.proposals``
    schema (see ``_schemas.py``); items inside ``interrupts`` match
    ``LibrarianInterrupt``.
    """

    session_id: str
    proposals: List[Dict[str, Any]]
    interrupts: List[Dict[str, Any]]


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
    """
    session_id = str(buffer_snapshot.get("session_id") or "?")

    record = runs.InvocationRecord(role="librarian", session_id=session_id)
    packet: EvidencePacket = _empty_packet(session_id)

    try:
        # ── Step 1: flatten the buffer ─────────────────────────────
        formatted_buffer, flat = lp.buffer_to_prompt_text(buffer_snapshot)
        record.input_buffer_turns = len(buffer_snapshot.get("turns") or [])

        # If the buffer has no quotable text at all, short-circuit.
        if not flat:
            log.debug(
                "librarian.scan: session=%s has no quotable turns; "
                "skipping LLM call",
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
        # No regex pre-pass, no BM25 seed hits. The Librarian has
        # Read/Glob tools and a library snapshot — it decides what to
        # inspect for dedup itself. Smart model, not pattern matching.
        kdir = knowledge_dir()
        library_snapshot = lp.build_library_snapshot(kdir)
        applies_when_table = lp.build_applies_when_table(kdir)
        slug = lp.derive_project_slug(buffer_snapshot)

        # ── Step 3: assemble + invoke ──────────────────────────────
        prompt = lp.assemble_prompt(
            project_slug=slug,
            rolling_buffer=formatted_buffer,
            library_snapshot=library_snapshot,
            applies_when_table=applies_when_table,
        )
        record.input_prompt = prompt

        log.debug(
            "librarian.scan: session=%s slug=%s turns=%d prompt_chars=%d",
            session_id, slug, len(flat), len(prompt),
        )

        # The Librarian needs cwd set to the knowledge dir so its
        # Read/Glob calls land in the right tree. ensure_home() creates
        # the dir if missing so we never pass a nonexistent path.
        ensure_home()
        try:
            response_text, cost_usd = llm.run_librarian_call(
                prompt, cwd=kdir, timeout_s=_LIBRARIAN_TIMEOUT_S
            )
        except llm.LLMTimeout as e:
            log.warning("librarian SDK timeout for session=%s: %s", session_id, e)
            record.mark_error(e)
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,

                "llm_timeout": 1,
            }
            return packet
        except Exception as e:
            log.exception("librarian SDK call failed for session=%s", session_id)
            record.mark_error(e)
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,

                "llm_error": 1,
            }
            return packet

        record.output_raw = response_text or ""
        record.cost_usd = float(cost_usd or 0.0)

        # ── Step 6: parse the JSON response ────────────────────────
        parsed = lp.parse_librarian_json(response_text)
        if parsed is None:
            log.warning(
                "librarian returned unparseable JSON (session=%s, chars=%d); "
                "emitting empty packet",
                session_id, len(response_text or ""),
            )
            record.parsed_ok = False
            record.decisions = {
                "proposals": 0,
                "interrupts": 0,

                "parse_failed": 1,
            }
            return packet

        normalised = lp.normalise_packet(parsed)
        packet = EvidencePacket(
            session_id=session_id,
            proposals=normalised["proposals"],
            interrupts=normalised["interrupts"],
        )
        record.parsed_ok = True
        record.decisions = {
            "proposals": len(normalised["proposals"]),
            "interrupts": len(normalised["interrupts"]),
        }
        log.info(
            "librarian.scan: session=%s emitted %d proposal(s), %d interrupt(s)",
            session_id,
            len(normalised["proposals"]),
            len(normalised["interrupts"]),
        )
        return packet

    except Exception as e:
        log.exception("librarian.scan unexpected error; returning empty packet")
        record.mark_error(e)
        return _empty_packet(session_id)

    finally:
        record.finalise()
