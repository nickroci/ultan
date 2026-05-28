"""Integrity-repair task queue — escalation bridge into the curator.

When the deterministic post-write pass (``scholar_prompt.repair_broken_wikilinks``)
finds an invariant violation it CANNOT fix on its own — today, a broken
wikilink with no unique on-disk target — it records a :class:`RepairTask`
here. The next Librarian run drains the pending tasks, renders them into
its prompt, and proposes an EXISTING curator action (rewrite the link,
write the missing target, or remove it with a reason). The Scholar
reviews/executes that proposal as a normal proposal. No new agent, no new
role — the same Librarian→Scholar pipeline, fed a different kind of input.

────────────────────────────────────────────────────────────────────
THE ONLY GUARD: in-flight, keyed on the issue fingerprint
────────────────────────────────────────────────────────────────────

The mandate is "re-escalate on EVERY detection while the issue is still
unfixed — no max-attempts cap, no permanent give-up." The single thing we
must never do is have two repair attempts for the SAME issue in flight at
once (that races the Scholar against itself). So:

  - :meth:`RepairQueue.enqueue` marks a fingerprint **in-flight** and
    queues a task. If the fingerprint is already in-flight, it returns
    ``False`` and queues nothing — the concurrent duplicate is dropped.
  - The marker STAYS set across :meth:`drain_pending` (the Librarian has
    taken the task but the attempt is not yet concluded).
  - :meth:`clear` removes the marker. It is called when the attempt
    concludes — when the Scholar finishes reviewing the packet that
    carried the task (proposal executed OR vetoed), and also defensively
    when a carrying packet is dropped before it reaches the Scholar
    (scheduler backpressure). Once cleared, the very next detection of a
    still-broken link re-escalates it.

The fingerprint is ``(kind, file, target)`` — stable across detections of
the same issue, distinct across different issues in the same file.

State is process-global and in-memory. A daemon restart wipes both the
pending queue and the in-flight markers; that is fine — nothing is stuck,
because a fresh deterministic pass re-detects any still-broken link and
re-escalates it. (For that to hold, the escalation path must NOT neutralise
the link it escalates — see ``scholar_prompt._repair_body_links``: an
escalated link is left broken on disk so it stays detectable until the
Scholar actually fixes it.)

Extensibility: :class:`RepairTask` carries a ``kind`` discriminator so
other invariant types (over-cap directories, malformed frontmatter) can
plug into the same queue + guard later. Only ``broken_wikilink`` is wired
today.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple, cast

# Fingerprint = (kind, file, target). A stable identity for "this exact
# issue" so repeated detections collapse onto one in-flight attempt.
Fingerprint = Tuple[str, str, str]

# v1 wires only this kind. New invariant types add their own literal and
# reuse the same enqueue/drain/clear machinery.
KIND_BROKEN_WIKILINK = "broken_wikilink"


@dataclass(frozen=True)
class RepairTask:
    """One integrity-repair task escalated to the Librarian.

    ``kind``    discriminator for the invariant type (``broken_wikilink``).
    ``file``    knowledge-root-relative path of the file holding the issue.
    ``target``  the offending value — for a wikilink, its broken target.
    ``context`` a short human-readable snippet so the Librarian can see
                where/how the issue appears without re-reading the file.
    """

    kind: str
    file: str
    target: str
    context: str = ""

    @property
    def fingerprint(self) -> Fingerprint:
        """Identity for the in-flight guard. ``context`` is excluded — the
        same broken link reported with slightly different surrounding text
        is still the SAME issue and must not double-escalate."""
        return (self.kind, self.file, self.target)


class RepairQueue:
    """Thread-safe pending-task queue with an in-flight fingerprint guard.

    All three operations take a single lock; the queue is small (one entry
    per unresolved invariant per pass) so contention is a non-issue.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: List[RepairTask] = []
        self._inflight: Set[Fingerprint] = set()

    def enqueue(self, task: RepairTask) -> bool:
        """Record a repair task unless its fingerprint is already in flight.

        Returns ``True`` when the task was queued (and the fingerprint
        marked in-flight), ``False`` when an attempt for the same
        fingerprint is already pending/in-flight — the concurrency guard
        rejecting a duplicate.
        """
        fp = task.fingerprint
        with self._lock:
            if fp in self._inflight:
                return False
            self._inflight.add(fp)
            self._pending.append(task)
            return True

    def drain_pending(self) -> List[RepairTask]:
        """Atomically remove and return all pending tasks.

        The returned fingerprints STAY in-flight — the caller (Librarian)
        has taken responsibility for the attempt but it is not concluded
        until the Scholar reviews the resulting packet and the caller
        calls :meth:`clear`.
        """
        with self._lock:
            drained = self._pending
            self._pending = []
            return drained

    def clear(self, fingerprints: List[Fingerprint]) -> None:
        """Release in-flight markers so a still-broken issue can re-escalate.

        Called when an escalation attempt concludes. Discarding a
        fingerprint that is not in-flight is a no-op, so double-clearing
        (e.g. backpressure-drop then a late Scholar pass) is safe.
        """
        if not fingerprints:
            return
        with self._lock:
            for fp in fingerprints:
                self._inflight.discard(fp)

    def inflight_count(self) -> int:
        """Number of fingerprints currently in flight (introspection/tests)."""
        with self._lock:
            return len(self._inflight)

    def pending_count(self) -> int:
        """Number of tasks waiting to be drained (introspection/tests)."""
        with self._lock:
            return len(self._pending)


# ── Process-global singleton ──────────────────────────────────────────
#
# The detection point (Scholar's repair pass) and the routing point
# (Librarian.scan) live in different modules and run on different threads;
# they coordinate through this one shared instance. A function accessor —
# not a module-level constant — so tests can reset state between cases via
# ``reset_queue()`` without re-importing.

_queue_lock = threading.Lock()
_queue_singleton: RepairQueue | None = None


def get_queue() -> RepairQueue:
    """Return the process-global :class:`RepairQueue`, creating it once."""
    global _queue_singleton
    with _queue_lock:
        if _queue_singleton is None:
            _queue_singleton = RepairQueue()
        return _queue_singleton


def reset_queue() -> None:
    """Drop the singleton so the next :func:`get_queue` builds a fresh one.

    Intended for tests that need isolated queue state between cases.
    """
    global _queue_singleton
    with _queue_lock:
        _queue_singleton = None


def fingerprints_of(tasks: List[RepairTask]) -> List[Fingerprint]:
    """Map tasks to their fingerprints — convenience for the caller that
    attaches them to an EvidencePacket."""
    return [t.fingerprint for t in tasks]


def parse_fingerprints(raw: object) -> List[Fingerprint]:
    """Coerce a packet's ``repair_fingerprints`` value back to a list of
    ``(kind, file, target)`` tuples, tolerating the wider runtime shape
    (the scheduler passes packets through as plain dicts; JSON round-trips
    turn tuples into lists). Anything malformed is skipped.

    Shared by the Scholar (release on review conclusion) and the scheduler
    (release on backpressure drop) so the parsing lives in exactly one
    place."""
    if not isinstance(raw, list):
        return []
    items = cast(List[object], raw)
    out: List[Fingerprint] = []
    for item in items:
        if not isinstance(item, (list, tuple)):
            continue
        seq = cast("Sequence[object]", item)
        if len(seq) != 3 or not all(isinstance(x, str) for x in seq):
            continue
        out.append((str(seq[0]), str(seq[1]), str(seq[2])))
    return out
