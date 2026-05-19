"""Scheduler — concurrent worker model wiring buffer events to
Librarian / Scholar invocations.

v2 architecture (this file). v1 was a single-threaded
``poll → tick`` loop; the Librarian's 30-60 s SDK call blocked the
Scholar trigger and starved the system under bursty traffic. This
module replaces that with four concurrent components, all stdlib
``threading`` / ``queue`` / ``concurrent.futures``:

  ``TailerThread``       wraps :class:`ingest.JsonlTailer.poll_once` on
                         its own thread; drops :class:`buffer.Event`s
                         into the buffer.
  ``DebounceScheduler``  one thread per scheduler instance. Watches
                         per-session inactivity timers; when a session
                         goes quiet for ``librarian_debounce_secs`` it
                         enqueues exactly one snapshot for the
                         Librarian pool. Repeated Stops within the
                         window collapse into a single Librarian run.
                         ``SessionEnd`` uses a shorter window
                         (``session_end_debounce_secs``) since the
                         session is over.
  ``LibrarianPool``      :class:`concurrent.futures.ThreadPoolExecutor`
                         of size ``librarian_concurrency`` (default 2).
                         Pulls snapshots from the librarian queue and
                         calls :func:`llm.run_librarian_call` (via the
                         injected ``librarian`` callable). Emits
                         :class:`librarian.EvidencePacket`s onto the
                         scholar queue.
  ``ScholarWorker``      one thread. Batches packets from the scholar
                         queue. Fires :func:`llm.run_scholar_call` (via
                         the injected ``scholar`` callable) when batch
                         threshold hit (K packets) OR T seconds elapsed
                         since last drain.

Backpressure: both internal queues are bounded (:class:`queue.Queue`,
default capacity 100). A producer that finds the queue full drops the
item with a WARN and bumps the corresponding counter. The rolling
buffer still owns the underlying events; we lose one Librarian /
Scholar pass on a backlog spike, not data.

Discipline mirrored from ``prior-art/claude-hooks/claude_hooks/_parallel.py``:
exceptions inside a worker thread MUST NOT take other workers down.
Every callable is wrapped in try/except + ``log.exception`` so a bad
proposal or a transient SDK error never cascades.

Shutdown: :meth:`Scheduler.stop` flips the stop event, waits for
in-flight work to drain (no SIGKILL of the SDK calls — they finish or
hit their own timeout), then forces one final Scholar drain so nothing
queued is lost.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .buffer import RollingBuffer
from . import librarian as librarian_mod
from . import scholar as scholar_mod
from .librarian import EvidencePacket


log = logging.getLogger("agent_mem_daemon.scheduler")


# Defaults — see scope brief / __main__.py for CLI knobs.
DEFAULT_LIBRARIAN_CONCURRENCY = 2     # parallel Haiku calls
DEFAULT_LIBRARIAN_DEBOUNCE_SECS = 30.0  # collapse bursty Stops
DEFAULT_SESSION_END_DEBOUNCE_SECS = 5.0  # SessionEnd: session is over
DEFAULT_SCHOLAR_EVERY_K = 3           # Scholar runs every K packets...
DEFAULT_SCHOLAR_EVERY_M_SECS = 60.0   # ...or every M seconds, whichever first
DEFAULT_SCHOLAR_MAX_BATCH = 10        # cap batch size for runaway bursts
DEFAULT_QUEUE_CEILING = 100           # bounded queue capacity
DEFAULT_SWEEP_INTERVAL_SECS = 300.0   # buffer.sweep cadence


LibrarianFn = Callable[[Dict[str, Any]], EvidencePacket]
ScholarFn = Callable[[List[EvidencePacket]], None]


@dataclass
class SchedulerConfig:
    librarian_concurrency: int = DEFAULT_LIBRARIAN_CONCURRENCY
    librarian_debounce_secs: float = DEFAULT_LIBRARIAN_DEBOUNCE_SECS
    session_end_debounce_secs: float = DEFAULT_SESSION_END_DEBOUNCE_SECS
    scholar_every_k: int = DEFAULT_SCHOLAR_EVERY_K
    scholar_every_m_secs: float = DEFAULT_SCHOLAR_EVERY_M_SECS
    scholar_max_batch: int = DEFAULT_SCHOLAR_MAX_BATCH
    queue_ceiling: int = DEFAULT_QUEUE_CEILING
    sweep_interval_secs: float = DEFAULT_SWEEP_INTERVAL_SECS


@dataclass
class SchedulerStats:
    """Counters useful for the doctor command and tests."""

    librarian_runs: int = 0
    librarian_skipped_backpressure: int = 0
    librarian_debounced: int = 0
    scholar_runs: int = 0
    scholar_skipped_backpressure: int = 0
    last_scholar_run_ts: float = 0.0
    last_sweep_ts: float = 0.0
    queue_high_water: int = 0
    packets_queued_total: int = 0
    packets_drained_total: int = 0


# ---------------------------------------------------------------------- #
# Debounce scheduler
# ---------------------------------------------------------------------- #


class DebounceScheduler:
    """Single thread watching per-session debounce timers.

    Maintains a dict of ``{session_id: scheduled_fire_at_monotonic}``.
    Sleeps until the next fire-time and then drains every entry whose
    fire-time has passed, handing the session id off to the supplied
    ``on_fire`` callback. Cheap (one thread for N sessions), simple
    (no per-session threads), correct under bursty input (each new
    arming call overwrites the existing scheduled time).
    """

    def __init__(self, *, on_fire: Callable[[str], None], name: str = "debounce"):
        self._on_fire = on_fire
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def arm(self, session_id: str, *, delay_secs: float) -> None:
        """Schedule (or re-schedule) a fire for ``session_id`` ``delay_secs``
        from now. If a timer already exists, the *later* arming wins —
        Stop bursts therefore extend the window the way the brief
        requires."""
        fire_at = time.monotonic() + max(delay_secs, 0.0)
        with self._cond:
            existing = self._timers.get(session_id)
            # The brief says: "If another Stop lands before the timer
            # fires, reset (extend) the timer." We honour that by
            # taking the max of (existing, new) — never shortening a
            # window that's already been set, only pushing it out.
            # SessionEnd's short window does *not* override a longer
            # Stop-driven window because SessionEnd merely shortens —
            # but the brief says SessionEnd shortens to 5 s, so we
            # explicitly let SessionEnd undercut by taking the *min*
            # via a separate code path below. ``arm`` therefore
            # always pushes the timer to fire_at (overwrite), and
            # ``arm_shorter`` is the dedicated "session is over" path.
            self._timers[session_id] = fire_at
            self._cond.notify_all()

    def arm_shorter(self, session_id: str, *, delay_secs: float) -> None:
        """Like :meth:`arm` but never pushes the timer further out.

        Used for ``SessionEnd``: we want the *shorter* of the existing
        and the new fire-time so a Stop's 30 s debounce doesn't keep
        an already-ended session waiting.
        """
        fire_at = time.monotonic() + max(delay_secs, 0.0)
        with self._cond:
            existing = self._timers.get(session_id)
            if existing is None or fire_at < existing:
                self._timers[session_id] = fire_at
                self._cond.notify_all()

    def cancel(self, session_id: str) -> None:
        with self._cond:
            self._timers.pop(session_id, None)
            self._cond.notify_all()

    def pending(self) -> int:
        with self._lock:
            return len(self._timers)

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        # Wait briefly for the thread to acknowledge — non-fatal if it
        # doesn't (daemon=True means the interpreter will reap it).
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                if not self._timers:
                    # Nothing scheduled — wait until either someone
                    # arms a timer or shutdown is requested.
                    self._cond.wait(timeout=1.0)
                    continue
                now = time.monotonic()
                # Next fire-time across all sessions.
                next_fire = min(self._timers.values())
                wait_for = next_fire - now
                if wait_for > 0:
                    # Sleep until either the next fire-time, a new arm
                    # call (notify), or shutdown.
                    self._cond.wait(timeout=wait_for)
                    continue
                # Collect everyone due.
                due = [
                    sid for sid, fire_at in self._timers.items()
                    if fire_at <= now
                ]
                for sid in due:
                    self._timers.pop(sid, None)
            # Fire outside the lock — the callback enqueues into another
            # bounded queue and we don't want to hold our cond while
            # that potentially blocks.
            for sid in due:
                try:
                    self._on_fire(sid)
                except Exception:
                    log.exception("debounce on_fire raised for session=%s", sid)


# ---------------------------------------------------------------------- #
# Scholar worker
# ---------------------------------------------------------------------- #


class ScholarWorker:
    """Single thread that drains the Scholar queue in batches.

    Trigger rule (matches scope brief §3):
      - Fire when ``queue_size >= scholar_every_k``, OR
      - Fire when ``scholar_every_m_secs`` elapsed since last drain
        and at least one packet is queued.
      - Cap batch size at ``scholar_max_batch`` so a runaway burst
        doesn't produce a multi-MB prompt.
      - If the queue is empty, wait on the condition (no busy poll).
    """

    def __init__(
        self,
        *,
        in_queue: "queue.Queue[EvidencePacket]",
        scholar_fn: ScholarFn,
        every_k: int,
        every_m_secs: float,
        max_batch: int,
        stats: SchedulerStats,
    ):
        self._q = in_queue
        self._fn = scholar_fn
        self._every_k = max(1, every_k)
        self._every_m_secs = max(0.1, every_m_secs)
        self._max_batch = max(1, max_batch)
        self._stats = stats
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="scholar-worker", daemon=True,
        )
        self._last_drain = time.monotonic()
        # Used by ``force_drain`` to ensure the final shutdown batch is
        # never partial just because the worker is mid-wait.
        self._drain_now = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, final_drain: bool = True) -> None:
        """Stop the worker. If ``final_drain``, flush whatever's queued
        before the thread exits."""
        if final_drain:
            self._drain_now.set()
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=30.0)

    def notify(self) -> None:
        """Wake the worker — a packet was just enqueued."""
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Decide whether to fire.
            qsize = self._q.qsize()
            elapsed = time.monotonic() - self._last_drain
            count_due = qsize >= self._every_k
            time_due = elapsed >= self._every_m_secs and qsize > 0
            if not (count_due or time_due):
                # Sleep until either a packet arrives (notify) or the
                # time-trigger could fire. We cap the wait at the time
                # threshold so we don't oversleep an M-secs trigger.
                wait_for = max(0.1, self._every_m_secs - elapsed)
                # When the queue is empty we don't have a useful time
                # bound — wait on the event with a longer slice.
                if qsize == 0:
                    wait_for = min(wait_for, 1.0)
                self._wake.wait(timeout=wait_for)
                self._wake.clear()
                continue

            self._drain_once()

        if self._drain_now.is_set():
            # Final shutdown drain.
            while not self._q.empty():
                self._drain_once()

    def _drain_once(self) -> None:
        batch: List[EvidencePacket] = []
        try:
            # Pull greedily up to max_batch.
            for _ in range(self._max_batch):
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if not batch:
                self._last_drain = time.monotonic()
                return
            log.info(
                "scholar-worker: draining batch of %d packet(s) "
                "(queue_remaining=%d)",
                len(batch), self._q.qsize(),
            )
            try:
                self._fn(batch)
            except Exception:
                log.exception(
                    "scholar callable raised; batch of %d dropped",
                    len(batch),
                )
            self._stats.scholar_runs += 1
            self._stats.packets_drained_total += len(batch)
            self._stats.last_scholar_run_ts = time.time()
        finally:
            self._last_drain = time.monotonic()


# ---------------------------------------------------------------------- #
# Tailer thread
# ---------------------------------------------------------------------- #


class TailerThread:
    """Wrap :class:`ingest.JsonlTailer.poll_once` in a thread.

    We don't use :meth:`JsonlTailer.run_forever` because we want the
    inter-poll interval to be a Scheduler concern (the tailer's own
    ``stop_event`` belongs to the daemon, not the scheduler).
    """

    def __init__(
        self,
        *,
        tailer,
        poll_interval: float,
        stop_event: threading.Event,
        name: str = "tailer",
    ):
        self._tailer = tailer
        self._poll_interval = poll_interval
        self._stop = stop_event
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        log.info("tailer thread: starting (poll=%.2fs)", self._poll_interval)
        while not self._stop.is_set():
            try:
                self._tailer.poll_once()
            except Exception:
                log.exception("tailer.poll_once raised; continuing")
            self._stop.wait(self._poll_interval)
        log.info("tailer thread: stopped")


# ---------------------------------------------------------------------- #
# Scheduler — coordinator
# ---------------------------------------------------------------------- #


class Scheduler:
    """Threaded coordinator.

    The previous synchronous Scheduler exposed ``on_event``, ``tick``,
    ``force_scholar``, ``queue_size``. We keep the same public surface
    so existing tests and callers continue to work; ``tick`` is now a
    no-op (kept for backwards-compat) because the timing decisions live
    in the debounce thread + scholar worker.
    """

    def __init__(
        self,
        *,
        buffer: RollingBuffer,
        config: Optional[SchedulerConfig] = None,
        librarian: LibrarianFn = librarian_mod.scan,
        scholar: ScholarFn = scholar_mod.review,
    ) -> None:
        self.buffer = buffer
        self.config = config or SchedulerConfig()
        self._librarian = librarian
        self._scholar = scholar
        self.stats = SchedulerStats()

        # Bounded queues — capacity == queue_ceiling, per scope brief.
        # We use two separate queues so a slow Scholar doesn't choke
        # the librarian pool's inputs (the librarian pool reads from
        # ``_librarian_queue``, writes to ``_scholar_queue``).
        cap = max(1, self.config.queue_ceiling)
        self._librarian_queue: "queue.Queue[str]" = queue.Queue(maxsize=cap)
        self._scholar_queue: "queue.Queue[EvidencePacket]" = queue.Queue(
            maxsize=cap,
        )

        # Debounce timer thread — arms one timer per session.
        self._debounce = DebounceScheduler(on_fire=self._on_debounce_fire)

        # Librarian thread pool (sized small — 2 by default).
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, self.config.librarian_concurrency),
            thread_name_prefix="librarian",
        )

        # Scholar worker.
        self._scholar_worker = ScholarWorker(
            in_queue=self._scholar_queue,
            scholar_fn=self._scholar,
            every_k=self.config.scholar_every_k,
            every_m_secs=self.config.scholar_every_m_secs,
            max_batch=self.config.scholar_max_batch,
            stats=self.stats,
        )

        # Buffer-sweep timer (we run it from a tiny dedicated thread to
        # keep the cadence honest under load).
        self._sweep_stop = threading.Event()
        self._sweep_thread: Optional[threading.Thread] = None

        # In-flight librarian futures — we wait on these at shutdown so
        # an SDK call mid-flight isn't cut off.
        self._in_flight_lock = threading.Lock()
        self._in_flight: set = set()

        self._started = False

    # ---- lifecycle --------------------------------------------------

    def start(self) -> None:
        """Spin up the debounce thread, scholar worker, and sweep
        thread. The librarian pool is created lazily by
        :class:`ThreadPoolExecutor` — work submitted before ``start``
        will still run, but ``start`` is the canonical entry point."""
        if self._started:
            return
        self._started = True
        self._debounce.start()
        self._scholar_worker.start()
        self._sweep_thread = threading.Thread(
            target=self._sweep_loop, name="buffer-sweep", daemon=True,
        )
        self._sweep_thread.start()
        log.info(
            "scheduler started (librarian_concurrency=%d "
            "librarian_debounce=%.1fs scholar_k=%d scholar_m=%.1fs "
            "scholar_max_batch=%d queue_ceiling=%d)",
            self.config.librarian_concurrency,
            self.config.librarian_debounce_secs,
            self.config.scholar_every_k,
            self.config.scholar_every_m_secs,
            self.config.scholar_max_batch,
            self.config.queue_ceiling,
        )

    def stop(self) -> None:
        """Clean shutdown:
          1. Stop the debounce thread (no new librarian work).
          2. Wait for in-flight librarian futures to complete (the SDK
             call has its own timeout — we don't kill it).
          3. Stop the scholar worker, letting it drain whatever's queued.
          4. Stop the sweep thread.
        """
        if not self._started:
            return
        log.info("scheduler stopping (graceful drain)")
        # 1. Stop debouncer first so no new sessions get enqueued.
        self._debounce.stop()

        # 2. Wait for in-flight librarian work. We deliberately don't
        # call ``ThreadPoolExecutor.shutdown(wait=True)`` until after
        # we've drained the librarian queue — there may be queued snapshots
        # that haven't been picked up by a worker yet but are needed by
        # downstream Scholar work.
        # Drain anything still queued by submitting it now (the workers
        # will pick it up).
        # Note: the librarian queue is drained by ``_librarian_worker``
        # naturally; we just need to make sure those running workers
        # finish. shutdown(wait=True) does both.
        try:
            self._pool.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            # Python < 3.9 doesn't support ``cancel_futures``.
            self._pool.shutdown(wait=True)

        # 3. Scholar worker: final drain.
        self._scholar_worker.stop(final_drain=True)

        # 4. Sweep thread.
        self._sweep_stop.set()
        if self._sweep_thread is not None:
            self._sweep_thread.join(timeout=2.0)
        log.info(
            "scheduler stopped: librarian_runs=%d backpressure_skips=%d "
            "scholar_runs=%d packets_queued=%d packets_drained=%d "
            "queue_high_water=%d librarian_debounced=%d "
            "scholar_backpressure_skips=%d",
            self.stats.librarian_runs,
            self.stats.librarian_skipped_backpressure,
            self.stats.scholar_runs,
            self.stats.packets_queued_total,
            self.stats.packets_drained_total,
            self.stats.queue_high_water,
            self.stats.librarian_debounced,
            self.stats.scholar_skipped_backpressure,
        )

    # ---- event-driven entry point ----------------------------------

    def on_event(self, ev) -> None:
        """Fold an Event into the buffer; arm the debounce timer when
        the event seals a turn."""
        sealed_sess = self.buffer.ingest(ev)
        if sealed_sess is None:
            return
        # Pick the appropriate debounce window.
        if ev.type == "SessionEnd":
            # Use ``arm_shorter`` so a prior Stop-driven 30 s window
            # doesn't make us wait an extra ~25 s on a session that's
            # already over.
            self._debounce.arm_shorter(
                sealed_sess.session_id,
                delay_secs=self.config.session_end_debounce_secs,
            )
            log.debug(
                "scheduler: armed SessionEnd debounce (session=%s "
                "delay=%.1fs)",
                sealed_sess.session_id,
                self.config.session_end_debounce_secs,
            )
        else:
            self._debounce.arm(
                sealed_sess.session_id,
                delay_secs=self.config.librarian_debounce_secs,
            )
            log.debug(
                "scheduler: armed Librarian debounce (session=%s "
                "delay=%.1fs)",
                sealed_sess.session_id,
                self.config.librarian_debounce_secs,
            )
            # Every Stop arming after the first within the window is a
            # debounced (collapsed) event — useful counter for
            # observability.
            self.stats.librarian_debounced += 1
        # ``needs_librarian`` is cleared once the snapshot is enqueued.

    # ---- debounce → librarian queue --------------------------------

    def _on_debounce_fire(self, session_id: str) -> None:
        """Called by the DebounceScheduler thread when a session's
        debounce window expires. Pushes a single snapshot onto the
        librarian queue and clears ``needs_librarian``."""
        sess = self.buffer.session(session_id)
        if sess is None:
            log.debug(
                "debounce fired for %s but session no longer present "
                "(swept?); skipping",
                session_id,
            )
            return
        try:
            self._librarian_queue.put(session_id, block=False)
        except queue.Full:
            self.stats.librarian_skipped_backpressure += 1
            log.warning(
                "BACKPRESSURE: librarian queue full (cap=%d); dropping "
                "session=%s — buffer keeps the events, we lose this pass",
                self.config.queue_ceiling, session_id,
            )
            return
        sess.needs_librarian = False
        # Submit a worker to consume this entry. We use a small
        # threadpool (size = librarian_concurrency) so concurrent
        # debounce fires for different sessions process in parallel.
        try:
            fut = self._pool.submit(self._librarian_worker)
        except RuntimeError:
            # Pool already shut down (mid-shutdown race). Drop quietly.
            log.debug("librarian pool shut down; dropping queued snapshot")
            return
        with self._in_flight_lock:
            self._in_flight.add(fut)
        fut.add_done_callback(self._on_librarian_done)

    def _on_librarian_done(self, fut) -> None:
        with self._in_flight_lock:
            self._in_flight.discard(fut)

    # ---- librarian worker ------------------------------------------

    def _librarian_worker(self) -> None:
        """One unit of work: pull one session_id off the queue, snapshot
        the buffer, run the Librarian, push the packet onto the Scholar
        queue. Exceptions are logged but never propagated — the
        ThreadPoolExecutor would otherwise eat them silently."""
        try:
            session_id = self._librarian_queue.get(timeout=1.0)
        except queue.Empty:
            return
        snap = self.buffer.snapshot(session_id)
        if snap is None:
            log.debug(
                "librarian worker: snapshot vanished for session=%s "
                "(swept between debounce fire and pickup)",
                session_id,
            )
            return
        try:
            packet = self._librarian(snap)
        except Exception:
            log.exception(
                "librarian callable raised for session=%s; dropping pass",
                session_id,
            )
            return
        if not isinstance(packet, dict):
            log.warning(
                "librarian returned non-dict %r; ignoring (session=%s)",
                type(packet), session_id,
            )
            return
        # Normalise — keep parity with v1 contract for downstream Scholar.
        packet.setdefault("session_id", session_id)
        packet.setdefault("proposals", packet.get("candidates", []) or [])
        packet.setdefault("interrupts", [])
        self.stats.librarian_runs += 1
        # Enqueue for the scholar.
        try:
            self._scholar_queue.put(packet, block=False)
        except queue.Full:
            self.stats.scholar_skipped_backpressure += 1
            log.warning(
                "BACKPRESSURE: scholar queue full (cap=%d); dropping "
                "packet for session=%s — Librarian pass wasted",
                self.config.queue_ceiling, session_id,
            )
            return
        self.stats.packets_queued_total += 1
        self.stats.queue_high_water = max(
            self.stats.queue_high_water,
            self._scholar_queue.qsize(),
        )
        # Wake the scholar so it can re-evaluate triggers.
        self._scholar_worker.notify()
        log.debug(
            "librarian worker: emitted packet for session=%s "
            "(scholar_queue=%d, runs=%d)",
            session_id,
            self._scholar_queue.qsize(),
            self.stats.librarian_runs,
        )

    # ---- sweep loop -------------------------------------------------

    def _sweep_loop(self) -> None:
        while not self._sweep_stop.is_set():
            # Sleep in small slices for prompt shutdown.
            self._sweep_stop.wait(timeout=self.config.sweep_interval_secs)
            if self._sweep_stop.is_set():
                return
            try:
                dropped = self.buffer.sweep()
                if dropped:
                    log.info(
                        "sweep dropped %d idle session(s): %s",
                        len(dropped), dropped,
                    )
                self.stats.last_sweep_ts = time.time()
            except Exception:
                log.exception("buffer sweep raised; will retry next interval")

    # ---- introspection / back-compat -------------------------------

    def tick(self, *, now: Optional[float] = None) -> None:  # noqa: ARG002
        """No-op in v2; kept for backwards compatibility with callers
        that were written against the v1 single-threaded loop. All
        timing decisions now happen inside the dedicated threads
        (debounce, scholar worker, sweep)."""
        return None

    def queue_size(self) -> int:
        """Returns the number of Scholar-bound packets currently queued.
        Provided for backwards compatibility — v1 callers compared this
        against ``queue_ceiling``. With the new architecture both the
        librarian queue and the scholar queue are bounded; this method
        reports the scholar one since that's what v1 meant."""
        return self._scholar_queue.qsize()

    def force_scholar(self, *, now: Optional[float] = None) -> None:  # noqa: ARG002
        """Drain the Scholar queue synchronously. Called on shutdown
        and exposed for tests / operators that need a forced flush."""
        # Inline-drain on the caller's thread so the test/operator gets
        # deterministic completion. The worker loop also drains, but
        # under shutdown we may need to flush before stopping the
        # worker thread.
        batch: List[EvidencePacket] = []
        try:
            while True:
                try:
                    batch.append(self._scholar_queue.get_nowait())
                except queue.Empty:
                    break
                if len(batch) >= self.config.scholar_max_batch:
                    # Process this batch and continue draining.
                    self._invoke_scholar_drain(batch)
                    batch = []
            if batch:
                self._invoke_scholar_drain(batch)
        except Exception:
            log.exception("force_scholar raised; partial drain only")

    def _invoke_scholar_drain(self, batch: List[EvidencePacket]) -> None:
        try:
            self._scholar(batch)
        except Exception:
            log.exception(
                "scholar callable raised in force_scholar; batch of %d lost",
                len(batch),
            )
        self.stats.scholar_runs += 1
        self.stats.packets_drained_total += len(batch)
        self.stats.last_scholar_run_ts = time.time()
