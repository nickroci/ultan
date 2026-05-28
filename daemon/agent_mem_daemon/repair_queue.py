"""Integrity-repair task queue — escalation bridge into the curator.

When a post-write check finds an invariant violation that cannot be fixed
deterministically, it records a :class:`RepairTask` here. The next Librarian
run drains the pending tasks, renders them into its prompt, and proposes an
EXISTING curator action; the Scholar reviews/executes it as a normal
proposal. No new agent, no new role — the same Librarian→Scholar pipeline,
fed a different kind of input. Three invariant kinds escalate through this
one mechanism (see the ``KIND_*`` constants), dispatched by ``kind``:

  - ``broken_wikilink`` — surfaced by ``scholar_prompt.repair_broken_wikilinks``
    when a link has no unique on-disk target → Librarian rewrites / writes
    the missing target / removes the link.
  - ``overcap_dir``     — surfaced by ``check_invariants`` when a flat dir
    exceeds the entry cap → Librarian proposes a ``split_folder`` (or
    ``move_entry``) to rebalance.
  - ``bad_frontmatter`` — surfaced by ``check_invariants`` when an entry's
    frontmatter is missing/unparseable/incomplete → Librarian proposes an
    ``update_entry`` that re-serialises valid frontmatter.

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
Scholar actually fixes it. The over-cap and bad-frontmatter checks are
already non-destructive — they only read — so re-detection is automatic.)

Generality: :class:`RepairTask` carries a ``kind`` discriminator and the
queue treats every kind identically. Adding a fourth invariant type is a
matter of (a) a new ``KIND_*`` literal, (b) a detection point that enqueues
it, and (c) a prompt section telling the Librarian how to repair it — no
change to the queue itself.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple, cast

# Fingerprint = (kind, file, target). A stable identity for "this exact
# issue" so repeated detections collapse onto one in-flight attempt.
Fingerprint = Tuple[str, str, str]

# Invariant-type discriminators. Every kind reuses the SAME
# enqueue/drain/clear machinery + in-flight guard — the queue itself stays
# kind-agnostic; only the detection point (Scholar) and the rendering point
# (Librarian prompt) branch on the value.
#
#   - ``broken_wikilink`` — a wikilink with no unique on-disk target.
#   - ``overcap_dir``     — a flat directory over ``MAX_FLAT_DIR_ENTRIES``.
#   - ``bad_frontmatter`` — an entry whose frontmatter is missing,
#                           unparseable, or short the required fields.
KIND_BROKEN_WIKILINK = "broken_wikilink"
KIND_OVERCAP_DIR = "overcap_dir"
KIND_BAD_FRONTMATTER = "bad_frontmatter"


@dataclass(frozen=True)
class RepairTask:
    """One integrity-repair task escalated to the Librarian.

    ``kind``    discriminator for the invariant type (one of ``KIND_*``).
    ``file``    knowledge-root-relative path of the file (or directory, for
                ``overcap_dir``) holding the issue.
    ``target``  the offending value, interpreted per ``kind``: a wikilink's
                broken target; the directory path for an over-cap dir; the
                entry path for bad frontmatter. Always non-empty so the
                fingerprint is a stable identity.
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
