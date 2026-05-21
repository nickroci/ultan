"""Scheduler v2 tests — threaded worker model.

The scheduler now owns:
  - one DebounceScheduler thread (per-session inactivity timer);
  - a ThreadPoolExecutor of librarian workers;
  - one ScholarWorker thread.

Tests inject stub callables for the librarian / scholar to assert
call counts, ordering, and concurrency. Real LLM code is never
invoked. All assertions are racy-by-nature but the timing windows
are kept short enough to keep the suite fast (target: each test
under 2 s).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List

import pytest

from agent_mem_daemon.buffer import Event, RollingBuffer
from agent_mem_daemon.librarian import EvidencePacket
from agent_mem_daemon.scheduler import (
    DebounceScheduler,
    Scheduler,
    SchedulerConfig,
)

# ---- helpers --------------------------------------------------------


def _ev(ts: float, sid: str, typ: str) -> Event:
    return Event(ts=ts, session_id=sid, type=typ, cwd="/repo", payload={})


def _empty_packet(snap: Dict[str, Any]) -> EvidencePacket:
    return EvidencePacket(
        session_id=snap["session_id"],
        proposals=[],
        interrupts=[],
    )


def _proposal_packet(snap: Dict[str, Any]) -> EvidencePacket:
    return EvidencePacket(
        session_id=snap["session_id"],
        proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "test"}],
        interrupts=[],
    )


def _wait_for(condition, *, timeout: float = 2.0, interval: float = 0.02):
    """Poll ``condition`` until it returns truthy or ``timeout`` elapses.
    Raises ``AssertionError`` on timeout — keeps test failure messages
    pointing at the failing condition rather than a bare timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise AssertionError(
        f"condition not satisfied within {timeout:.2f}s "
        f"({getattr(condition, '__name__', condition)!r})"
    )


@pytest.fixture
def quick_config():
    """Tight timings so the test suite stays fast."""
    return SchedulerConfig(
        librarian_concurrency=2,
        librarian_debounce_secs=0.10,
        session_end_debounce_secs=0.02,
        scholar_every_k=3,
        scholar_every_m_secs=0.20,
        scholar_max_batch=10,
        queue_ceiling=100,
        sweep_interval_secs=999.0,  # disable for tests
    )


# ====================================================================
# DebounceScheduler unit tests
# ====================================================================


def test_debounce_collapses_rapid_arms_into_one_fire():
    """Five Stops within the debounce window should fire exactly once."""
    fires: List[str] = []
    db = DebounceScheduler(on_fire=lambda sid: fires.append(sid))
    db.start()
    try:
        # All five arms land within 50 ms — debounce window is 100 ms,
        # so each new arm pushes the timer to fire-at = now + 100 ms.
        for _ in range(5):
            db.arm("s1", delay_secs=0.10)
            time.sleep(0.01)
        # Wait long enough for the (extended) window to fire.
        _wait_for(lambda: len(fires) == 1, timeout=2.0)
        # Give it another beat to make sure nothing else fires.
        time.sleep(0.15)
        assert fires == ["s1"]
    finally:
        db.stop()


def test_debounce_separated_arms_fire_separately():
    """Arms spaced beyond the window each fire on their own."""
    fires: List[str] = []
    lock = threading.Lock()

    def _on_fire(sid):
        with lock:
            fires.append(sid)

    db = DebounceScheduler(on_fire=_on_fire)
    db.start()
    try:
        for _ in range(3):
            db.arm("s1", delay_secs=0.05)
            time.sleep(0.20)  # well beyond debounce window
        _wait_for(lambda: len(fires) == 3, timeout=2.0)
        assert fires == ["s1", "s1", "s1"]
    finally:
        db.stop()


def test_debounce_arm_shorter_undercuts_existing_timer():
    """``arm_shorter`` must never push the timer further out, only nearer."""
    fires: List[float] = []
    db = DebounceScheduler(on_fire=lambda sid: fires.append(time.monotonic()))
    db.start()
    started = time.monotonic()
    try:
        # Arm a long window first.
        db.arm("s1", delay_secs=2.0)
        # Immediately shorten via arm_shorter.
        db.arm_shorter("s1", delay_secs=0.05)
        _wait_for(lambda: len(fires) == 1, timeout=1.0)
        elapsed = fires[0] - started
        # Should fire well before the 2 s the long arm asked for.
        assert elapsed < 0.5, f"expected <0.5s, got {elapsed:.3f}s"
    finally:
        db.stop()


def test_debounce_arm_shorter_does_not_push_window_out():
    """If an existing timer is shorter, ``arm_shorter`` must not extend it."""
    fires: List[float] = []
    db = DebounceScheduler(on_fire=lambda sid: fires.append(time.monotonic()))
    db.start()
    started = time.monotonic()
    try:
        db.arm("s1", delay_secs=0.05)
        # arm_shorter with a longer delay should be a no-op.
        db.arm_shorter("s1", delay_secs=2.0)
        _wait_for(lambda: len(fires) == 1, timeout=1.0)
        elapsed = fires[0] - started
        assert elapsed < 0.5
    finally:
        db.stop()


# ====================================================================
# Scheduler integration tests
# ====================================================================


def test_debounce_collapses_repeated_stops(quick_config):
    """5 Stops on the same session inside the debounce window → exactly
    1 Librarian invocation."""
    buf = RollingBuffer()
    lib_calls: List[str] = []
    lib_lock = threading.Lock()

    def _lib(snap):
        with lib_lock:
            lib_calls.append(snap["session_id"])
        return _empty_packet(snap)

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_lib,
        scholar=lambda batch: None,
    )
    sched.start()
    try:
        for i in range(5):
            sched.on_event(_ev(float(i), "s1", "Stop"))
            time.sleep(0.01)  # well inside debounce window
        _wait_for(lambda: len(lib_calls) == 1, timeout=2.0)
        # Wait beyond the window to confirm nothing else fires.
        time.sleep(0.20)
        with lib_lock:
            assert lib_calls == ["s1"]
    finally:
        sched.stop()


def test_separated_stops_each_fire_librarian(quick_config):
    """Stops spaced beyond the debounce window each produce one Librarian
    invocation."""
    buf = RollingBuffer()
    lib_calls: List[str] = []
    lib_lock = threading.Lock()

    def _lib(snap):
        with lib_lock:
            lib_calls.append(snap["session_id"])
        return _empty_packet(snap)

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_lib,
        scholar=lambda batch: None,
    )
    sched.start()
    try:
        for i in range(3):
            sched.on_event(_ev(float(i), "s1", "Stop"))
            time.sleep(0.25)  # > debounce window of 0.10
        _wait_for(lambda: len(lib_calls) == 3, timeout=3.0)
    finally:
        sched.stop()


def test_session_end_uses_shorter_debounce(quick_config):
    """SessionEnd should fire fast (≤ session_end_debounce_secs * 5)
    even if a Stop earlier in the session armed a longer window."""
    buf = RollingBuffer()
    lib_calls: List[float] = []
    lib_lock = threading.Lock()

    def _lib(snap):
        with lib_lock:
            lib_calls.append(time.monotonic())
        return _empty_packet(snap)

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_lib,
        scholar=lambda batch: None,
    )
    sched.start()
    started = time.monotonic()
    try:
        sched.on_event(_ev(1.0, "s1", "Stop"))  # arms 0.10s window
        time.sleep(0.01)
        sched.on_event(_ev(2.0, "s1", "SessionEnd"))  # shortens to 0.02s
        _wait_for(lambda: len(lib_calls) == 1, timeout=1.0)
        # Should have fired close to session_end_debounce, well before
        # the 0.10s Stop window would have allowed.
        elapsed = lib_calls[0] - started
        assert elapsed < 0.08, f"SessionEnd should have shortened window; got {elapsed:.3f}s"
    finally:
        sched.stop()


def test_parallel_librarian_workers_run_concurrently(quick_config):
    """3 sessions firing simultaneously should process in parallel (with
    pool size ≥ 2). Wall-clock should be much closer to one Librarian's
    sleep than to 3× it."""
    buf = RollingBuffer()
    lib_lock = threading.Lock()
    in_flight = [0]
    peak_in_flight = [0]
    completed: List[str] = []

    def _slow_lib(snap):
        with lib_lock:
            in_flight[0] += 1
            peak_in_flight[0] = max(peak_in_flight[0], in_flight[0])
        try:
            time.sleep(0.30)
            return _empty_packet(snap)
        finally:
            with lib_lock:
                in_flight[0] -= 1
                completed.append(snap["session_id"])

    cfg = SchedulerConfig(
        librarian_concurrency=3,
        librarian_debounce_secs=0.05,
        session_end_debounce_secs=0.02,
        scholar_every_k=1000,
        scholar_every_m_secs=999.0,
        scholar_max_batch=10,
        queue_ceiling=100,
        sweep_interval_secs=999.0,
    )
    sched = Scheduler(
        buffer=buf,
        config=cfg,
        librarian=_slow_lib,
        scholar=lambda batch: None,
    )
    sched.start()
    started = time.monotonic()
    try:
        for sid in ("s1", "s2", "s3"):
            sched.on_event(_ev(1.0, sid, "Stop"))
        # Wait for all three to complete.
        _wait_for(lambda: len(completed) == 3, timeout=3.0)
        elapsed = time.monotonic() - started
        # Serial would be ≥ 3 × 0.30 = 0.90 s + debounce. Parallel
        # should be closer to 1 × 0.30 + debounce ≈ 0.40 s.
        assert elapsed < 0.80, (
            f"workers did not run in parallel; wall-clock={elapsed:.3f}s (serial would be ≥0.90s)"
        )
        assert peak_in_flight[0] >= 2, (
            f"expected ≥2 concurrent librarians, observed peak={peak_in_flight[0]}"
        )
    finally:
        sched.stop()


def test_scholar_batches_rapid_packets(quick_config):
    """K=3 packets in quick succession should produce ONE Scholar
    invocation with all 3 packets in the batch."""
    buf = RollingBuffer()
    sch_batches: List[List[Dict[str, Any]]] = []
    sch_lock = threading.Lock()

    def _sch(batch):
        with sch_lock:
            sch_batches.append(list(batch))

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_proposal_packet,
        scholar=_sch,
    )
    sched.start()
    try:
        # Three Stops on different sessions — each will get its own
        # Librarian pass after the debounce, producing 3 packets fast.
        for sid in ("s1", "s2", "s3"):
            sched.on_event(_ev(1.0, sid, "Stop"))
        # Wait for Scholar to fire.
        _wait_for(lambda: len(sch_batches) >= 1, timeout=3.0)
        # The first batch should hold all 3 packets.
        assert len(sch_batches[0]) == 3, (
            f"expected 3-packet batch, got sizes {[len(b) for b in sch_batches]}"
        )
    finally:
        sched.stop()


def test_scholar_fires_on_timer_for_single_packet(quick_config):
    """K=3, M=0.20s: a single packet should still fire the Scholar after
    M seconds elapsed since the last drain."""
    buf = RollingBuffer()
    sch_batches: List[int] = []
    sch_lock = threading.Lock()

    def _sch(batch):
        with sch_lock:
            sch_batches.append(len(batch))

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_proposal_packet,
        scholar=_sch,
    )
    sched.start()
    try:
        sched.on_event(_ev(1.0, "s1", "Stop"))
        # Wait for Librarian + Scholar timer to elapse.
        _wait_for(lambda: len(sch_batches) >= 1, timeout=2.0)
        assert sch_batches[0] == 1
    finally:
        sched.stop()


def test_scholar_max_batch_caps_burst(quick_config):
    """A burst of 15 packets with max_batch=5 should produce ≥ 3 batches,
    none larger than 5."""
    cfg = SchedulerConfig(
        librarian_concurrency=4,
        librarian_debounce_secs=0.05,
        session_end_debounce_secs=0.02,
        scholar_every_k=2,
        scholar_every_m_secs=0.10,
        scholar_max_batch=5,
        queue_ceiling=100,
        sweep_interval_secs=999.0,
    )
    buf = RollingBuffer()
    sch_batches: List[int] = []
    sch_lock = threading.Lock()

    def _sch(batch):
        with sch_lock:
            sch_batches.append(len(batch))

    sched = Scheduler(
        buffer=buf,
        config=cfg,
        librarian=_proposal_packet,
        scholar=_sch,
    )
    sched.start()
    try:
        for i in range(15):
            sched.on_event(_ev(float(i), f"s{i}", "Stop"))
        # All 15 packets must be drained eventually.
        _wait_for(
            lambda: sum(sch_batches) >= 15,
            timeout=5.0,
        )
        with sch_lock:
            sizes = list(sch_batches)
        assert all(s <= 5 for s in sizes), f"batch larger than cap: sizes={sizes}"
        assert sum(sizes) == 15
    finally:
        sched.stop()


def test_backpressure_drops_packet_on_full_queue(caplog):
    """When the scholar queue is at its ceiling, the librarian worker
    must drop new packets with a WARN rather than blocking."""
    cfg = SchedulerConfig(
        librarian_concurrency=2,
        librarian_debounce_secs=0.02,
        session_end_debounce_secs=0.02,
        scholar_every_k=100,  # don't drain
        scholar_every_m_secs=999.0,
        scholar_max_batch=10,
        queue_ceiling=2,  # tiny ceiling
        sweep_interval_secs=999.0,
    )
    buf = RollingBuffer()

    def _lib(snap):
        return _proposal_packet(snap)

    sch_calls: List[int] = []

    def _sch(batch):
        sch_calls.append(len(batch))

    sched = Scheduler(buffer=buf, config=cfg, librarian=_lib, scholar=_sch)
    sched.start()
    caplog.set_level(logging.WARNING)
    try:
        # 5 distinct sessions → 5 packets, queue cap is 2.
        for i in range(5):
            sched.on_event(_ev(float(i), f"s{i}", "Stop"))
        # Give the librarian pool time to run.
        _wait_for(
            lambda: sched.stats.scholar_skipped_backpressure >= 1,
            timeout=3.0,
        )
        # At least one packet got dropped.
        assert sched.stats.scholar_skipped_backpressure >= 1
        # WARN was logged.
        assert any("BACKPRESSURE" in r.message for r in caplog.records)
        # And no crash — scheduler still running.
    finally:
        sched.stop()


def test_shutdown_drains_queued_packets():
    """On stop(), any packets still sitting in the scholar queue must be
    flushed to the Scholar callable."""
    cfg = SchedulerConfig(
        librarian_concurrency=2,
        librarian_debounce_secs=0.02,
        session_end_debounce_secs=0.02,
        scholar_every_k=100,  # never fire on count
        scholar_every_m_secs=999.0,  # never fire on time
        scholar_max_batch=10,
        queue_ceiling=100,
        sweep_interval_secs=999.0,
    )
    buf = RollingBuffer()
    sch_batches: List[int] = []
    sch_lock = threading.Lock()

    def _sch(batch):
        with sch_lock:
            sch_batches.append(len(batch))

    sched = Scheduler(
        buffer=buf,
        config=cfg,
        librarian=_proposal_packet,
        scholar=_sch,
    )
    sched.start()
    try:
        for sid in ("s1", "s2", "s3", "s4"):
            sched.on_event(_ev(1.0, sid, "Stop"))
        # Wait for all 4 packets to land in the scholar queue (still
        # un-drained because thresholds are high).
        _wait_for(lambda: sched.queue_size() == 4, timeout=2.0)
        # The Scholar timer M is 999s and K is 100, so nothing has
        # fired yet.
        with sch_lock:
            assert sch_batches == []
    finally:
        sched.stop()
    # After stop(), the final drain must have flushed all 4 packets.
    with sch_lock:
        assert sum(sch_batches) == 4, f"final drain incomplete; batches={sch_batches}"


def test_librarian_exception_does_not_crash_pool(quick_config, caplog):
    """A librarian that raises must be logged and other librarian calls
    must continue to run."""
    buf = RollingBuffer()
    call_count = [0]
    successful: List[str] = []
    lock = threading.Lock()

    def _flaky(snap):
        with lock:
            call_count[0] += 1
            if snap["session_id"] == "bad":
                raise RuntimeError("boom")
            successful.append(snap["session_id"])
        return _empty_packet(snap)

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_flaky,
        scholar=lambda batch: None,
    )
    sched.start()
    caplog.set_level(logging.WARNING)
    try:
        for sid in ("good1", "bad", "good2"):
            sched.on_event(_ev(1.0, sid, "Stop"))
        _wait_for(lambda: len(successful) == 2, timeout=3.0)
        assert set(successful) == {"good1", "good2"}
        # Exception was logged.
        assert any("librarian callable raised" in r.message for r in caplog.records)
    finally:
        sched.stop()


def test_scholar_exception_does_not_crash_worker(quick_config, caplog):
    """A scholar that raises must not stop subsequent drains."""
    buf = RollingBuffer()
    sch_attempts = [0]
    lock = threading.Lock()

    def _flaky_sch(batch):
        with lock:
            sch_attempts[0] += 1
            if sch_attempts[0] == 1:
                raise RuntimeError("scholar boom")

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_proposal_packet,
        scholar=_flaky_sch,
    )
    sched.start()
    caplog.set_level(logging.WARNING)
    try:
        # First batch (≥3 packets) will raise.
        for sid in ("s1", "s2", "s3"):
            sched.on_event(_ev(1.0, sid, "Stop"))
        _wait_for(lambda: sch_attempts[0] >= 1, timeout=3.0)
        # Second batch should still go through.
        time.sleep(0.05)
        for sid in ("t1", "t2", "t3"):
            sched.on_event(_ev(2.0, sid, "Stop"))
        _wait_for(lambda: sch_attempts[0] >= 2, timeout=3.0)
        assert any("scholar callable raised" in r.message for r in caplog.records)
    finally:
        sched.stop()


def test_on_event_non_stop_does_not_arm_debounce(quick_config):
    """PostToolUse and similar events don't seal a turn, so they must
    not arm a debounce timer."""
    buf = RollingBuffer()
    lib_calls: List[str] = []

    def _lib(snap):
        lib_calls.append(snap["session_id"])
        return _empty_packet(snap)

    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_lib,
        scholar=lambda batch: None,
    )
    sched.start()
    try:
        for i in range(5):
            sched.on_event(_ev(float(i), "s1", "PostToolUse"))
        time.sleep(0.30)  # well past debounce window
        assert lib_calls == []
    finally:
        sched.stop()


def test_queue_size_reports_scholar_queue(quick_config):
    """``queue_size`` is back-compat for callers that compared the
    pending-packet count against the ceiling."""
    cfg = SchedulerConfig(
        librarian_concurrency=2,
        librarian_debounce_secs=0.02,
        session_end_debounce_secs=0.02,
        scholar_every_k=100,
        scholar_every_m_secs=999.0,
        scholar_max_batch=10,
        queue_ceiling=100,
        sweep_interval_secs=999.0,
    )
    buf = RollingBuffer()
    sched = Scheduler(
        buffer=buf,
        config=cfg,
        librarian=_proposal_packet,
        scholar=lambda batch: None,
    )
    sched.start()
    try:
        for sid in ("s1", "s2"):
            sched.on_event(_ev(1.0, sid, "Stop"))
        _wait_for(lambda: sched.queue_size() == 2, timeout=2.0)
    finally:
        sched.stop()


def test_force_scholar_drains_immediately():
    """force_scholar() must flush the queue on the caller's thread,
    not wait for the worker."""
    cfg = SchedulerConfig(
        librarian_concurrency=2,
        librarian_debounce_secs=0.02,
        session_end_debounce_secs=0.02,
        scholar_every_k=100,
        scholar_every_m_secs=999.0,
        scholar_max_batch=10,
        queue_ceiling=100,
        sweep_interval_secs=999.0,
    )
    buf = RollingBuffer()
    sch_batches: List[int] = []

    def _sch(batch):
        sch_batches.append(len(batch))

    sched = Scheduler(
        buffer=buf,
        config=cfg,
        librarian=_proposal_packet,
        scholar=_sch,
    )
    sched.start()
    try:
        for sid in ("s1", "s2"):
            sched.on_event(_ev(1.0, sid, "Stop"))
        _wait_for(lambda: sched.queue_size() == 2, timeout=2.0)
        sched.force_scholar()
        # Sync drain — should have happened by now.
        assert sum(sch_batches) == 2
    finally:
        sched.stop()


def test_tick_is_noop_for_backcompat(quick_config):
    """``tick`` is a no-op in v2 but the symbol must still exist."""
    buf = RollingBuffer()
    sched = Scheduler(
        buffer=buf,
        config=quick_config,
        librarian=_empty_packet,
        scholar=lambda batch: None,
    )
    sched.tick()  # must not raise
    sched.tick(now=1234.0)
