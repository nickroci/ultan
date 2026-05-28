"""Tests for the integrity-repair queue + its in-flight concurrency guard.

The queue is the bridge that escalates an invariant the deterministic
post-write pass can't fix (today: an unresolvable wikilink) into the
Librarian→Scholar pipeline. Its ONE guard is the in-flight marker keyed
on the issue fingerprint ``(kind, file, target)``:

  - enqueue marks a fingerprint in-flight and queues a task;
  - a second enqueue for the SAME fingerprint while it's still in-flight
    is rejected (no duplicate, no second concurrent attempt);
  - the marker survives ``drain_pending`` (attempt taken, not concluded);
  - ``clear`` releases the marker so the very next detection re-escalates.

These are the invariants the whole escalation loop rests on, so they get
direct, fast unit coverage here; the end-to-end wiring is exercised in
``test_scholar.py`` / ``test_librarian.py``.
"""

from __future__ import annotations

import threading

from agent_mem_daemon import repair_queue


def _task(
    file: str = "global/python/x.md", target: str = "global/ghost/missing"
) -> repair_queue.RepairTask:
    return repair_queue.RepairTask(
        kind=repair_queue.KIND_BROKEN_WIKILINK,
        file=file,
        target=target,
        context="see [[...]] here",
    )


def test_enqueue_marks_inflight_and_queues() -> None:
    q = repair_queue.RepairQueue()
    assert q.enqueue(_task()) is True
    assert q.pending_count() == 1
    assert q.inflight_count() == 1


def test_duplicate_fingerprint_rejected_while_inflight() -> None:
    # The ONLY guard: a second enqueue for the same (kind, file, target)
    # is dropped while the first is still in flight — never two concurrent
    # attempts for one issue.
    q = repair_queue.RepairQueue()
    assert q.enqueue(_task()) is True
    assert q.enqueue(_task()) is False  # duplicate rejected
    assert q.pending_count() == 1
    assert q.inflight_count() == 1


def test_context_not_part_of_fingerprint() -> None:
    # The same broken link reported with different surrounding text is the
    # SAME issue and must not double-escalate.
    q = repair_queue.RepairQueue()
    a = repair_queue.RepairTask(
        kind=repair_queue.KIND_BROKEN_WIKILINK,
        file="f.md",
        target="t",
        context="context A",
    )
    b = repair_queue.RepairTask(
        kind=repair_queue.KIND_BROKEN_WIKILINK,
        file="f.md",
        target="t",
        context="completely different context B",
    )
    assert a.fingerprint == b.fingerprint
    assert q.enqueue(a) is True
    assert q.enqueue(b) is False


def test_different_targets_in_same_file_both_enqueue() -> None:
    q = repair_queue.RepairQueue()
    assert q.enqueue(_task(target="global/ghost/one")) is True
    assert q.enqueue(_task(target="global/ghost/two")) is True
    assert q.pending_count() == 2
    assert q.inflight_count() == 2


def test_drain_keeps_fingerprint_inflight() -> None:
    # Draining hands the task to the Librarian but does NOT conclude the
    # attempt — the marker stays set so a detection mid-attempt is still
    # suppressed.
    q = repair_queue.RepairQueue()
    q.enqueue(_task())
    drained = q.drain_pending()
    assert len(drained) == 1
    assert q.pending_count() == 0
    assert q.inflight_count() == 1  # still in flight
    # A detection after drain but before clear is still a duplicate.
    assert q.enqueue(_task()) is False


def test_clear_allows_reescalation() -> None:
    # Concluding the attempt releases the marker; the next detection of the
    # still-broken issue re-escalates. No max-attempts cap.
    q = repair_queue.RepairQueue()
    q.enqueue(_task())
    drained = q.drain_pending()
    q.clear(repair_queue.fingerprints_of(drained))
    assert q.inflight_count() == 0
    # Re-escalation now succeeds.
    assert q.enqueue(_task()) is True
    assert q.pending_count() == 1


def test_clear_of_unknown_fingerprint_is_noop() -> None:
    # Double-clear (e.g. backpressure drop then a late Scholar pass) must
    # be safe.
    q = repair_queue.RepairQueue()
    q.enqueue(_task())
    fps = repair_queue.fingerprints_of([_task()])
    q.clear(fps)
    q.clear(fps)  # second clear: no error, no negative count
    assert q.inflight_count() == 0


def test_clear_empty_list_noop() -> None:
    q = repair_queue.RepairQueue()
    q.clear([])  # must not raise
    assert q.inflight_count() == 0


def test_drain_empty_returns_empty() -> None:
    q = repair_queue.RepairQueue()
    assert q.drain_pending() == []


def test_parse_fingerprints_tolerates_shapes() -> None:
    # Tuples, lists (JSON round-trip), and malformed entries.
    raw = [
        ("broken_wikilink", "f.md", "t"),  # tuple
        ["broken_wikilink", "g.md", "u"],  # list
        ["too", "short"],  # wrong arity → skipped
        "not-a-seq",  # wrong type → skipped
        ["a", "b", 3],  # non-str element → skipped
    ]
    out = repair_queue.parse_fingerprints(raw)
    assert out == [
        ("broken_wikilink", "f.md", "t"),
        ("broken_wikilink", "g.md", "u"),
    ]


def test_parse_fingerprints_non_list_returns_empty() -> None:
    assert repair_queue.parse_fingerprints(None) == []
    assert repair_queue.parse_fingerprints("nope") == []
    assert repair_queue.parse_fingerprints(42) == []


def test_get_queue_is_singleton_and_resettable() -> None:
    repair_queue.reset_queue()
    a = repair_queue.get_queue()
    b = repair_queue.get_queue()
    assert a is b
    repair_queue.reset_queue()
    c = repair_queue.get_queue()
    assert c is not a


def test_enqueue_is_threadsafe_no_double_escalation() -> None:
    # Hammer enqueue from many threads with the SAME fingerprint: exactly
    # ONE must win (the guard holds under contention).
    q = repair_queue.RepairQueue()
    results: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    def worker() -> None:
        barrier.wait()  # maximise the race window
        ok = q.enqueue(_task())
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r) == 1  # exactly one escalation
    assert q.pending_count() == 1
    assert q.inflight_count() == 1
