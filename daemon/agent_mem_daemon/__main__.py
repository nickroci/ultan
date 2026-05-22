"""Daemon entry point.

Foreground-only v1 (matches scope brief). Backgrounding / launchd is a
Phase-4 task and intentionally left as a TODO so we don't pretend we've
shipped it.

Lifecycle:
  1. Parse args.
  2. Configure logging (rotated file + optional stderr).
  3. Acquire PID file (refuse to start if a live daemon already holds it).
  4. Install SIGTERM/SIGINT handlers that flip a stop event.
  5. Spin up the scheduler + tailer.
  6. Run until stopped, then drain the Scholar queue + clean up.
"""

from __future__ import annotations

import argparse
import errno
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

from .buffer import DEFAULT_INACTIVITY_SECONDS, DEFAULT_MAX_TURNS, RollingBuffer
from .ingest import DEFAULT_POLL_INTERVAL, JsonlTailer
from .logging_setup import configure as configure_logging
from .paths import (
    ensure_home,
    events_path,
    knowledge_dir,
    log_path,
    offset_state_path,
    pid_path,
)
from .priming_rpc import PrimingRpcThread
from .scheduler import (
    DEFAULT_LIBRARIAN_CONCURRENCY,
    DEFAULT_LIBRARIAN_DEBOUNCE_SECS,
    DEFAULT_QUEUE_CEILING,
    DEFAULT_SCHOLAR_EVERY_K,
    DEFAULT_SCHOLAR_EVERY_M_SECS,
    DEFAULT_SCHOLAR_MAX_BATCH,
    DEFAULT_SESSION_END_DEBOUNCE_SECS,
    DEFAULT_SWEEP_INTERVAL_SECS,
    Scheduler,
    SchedulerConfig,
    TailerThread,
)

# ---- PID file -------------------------------------------------------


def _read_pid(path: Path) -> int | None:
    try:
        txt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    """Cheap liveness check via signal 0. Returns True iff the process
    exists and we have permission to signal it. Permission denied still
    counts as alive — better safe than racing two daemons."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as e:
        # On some platforms EPERM surfaces as OSError with errno EPERM.
        return e.errno == errno.EPERM


def acquire_pid_file(path: Path) -> None:
    """Write our PID to ``path`` after verifying no live daemon already
    owns it. Raises SystemExit(2) on conflict."""
    existing = _read_pid(path)
    if existing is not None and _pid_alive(existing):
        sys.stderr.write(
            f"agent-mem-daemon: another daemon appears to be running "
            f"(PID {existing} from {path}). Refusing to start.\n"
        )
        raise SystemExit(2)
    # Stale PID file or no PID file — claim it.
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"agent-mem-daemon: cannot write PID file {path}: {e}\n")
        raise SystemExit(1)


def release_pid_file(path: Path) -> None:
    """Best-effort remove. Never raises."""
    try:
        current = _read_pid(path)
        if current == os.getpid():
            path.unlink()
    except OSError:
        pass


# ---- args -----------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-mem-daemon",
        description="Event-ingest daemon for agent-mem (skeleton phase, no LLM).",
    )
    p.add_argument(
        "--foreground",
        action="store_true",
        default=True,
        help="Run in the foreground (default; the only mode in v1). "
        "Kept as an explicit flag so future launchd/background "
        "modes can land without breaking invocations.",
    )
    p.add_argument(
        "--events-file",
        type=Path,
        default=None,
        help="Path to the JSONL events file (default: ~/.agent-mem/events.jsonl).",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to the daemon log (default: ~/.agent-mem/daemon.log).",
    )
    p.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="Path to the daemon PID file (default: ~/.agent-mem/daemon.pid).",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between file polls (default: {DEFAULT_POLL_INTERVAL}).",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Per-session rolling buffer size in turns (default: {DEFAULT_MAX_TURNS}).",
    )
    p.add_argument(
        "--inactivity-seconds",
        type=float,
        default=DEFAULT_INACTIVITY_SECONDS,
        help=f"Drop sessions idle longer than this (default: {DEFAULT_INACTIVITY_SECONDS}s).",
    )
    p.add_argument(
        "--librarian-concurrency",
        type=int,
        default=DEFAULT_LIBRARIAN_CONCURRENCY,
        help=(
            f"Parallel Librarian SDK calls (default: "
            f"{DEFAULT_LIBRARIAN_CONCURRENCY}). The Librarian's per-call "
            "wall time is dominated by network I/O; the GIL releases "
            "during socket reads so a small threadpool gives near-linear "
            "speedup without asyncio."
        ),
    )
    p.add_argument(
        "--librarian-debounce-secs",
        type=float,
        default=DEFAULT_LIBRARIAN_DEBOUNCE_SECS,
        help=(
            f"Per-session Stop debounce window (default: "
            f"{DEFAULT_LIBRARIAN_DEBOUNCE_SECS}s). Bursty sessions collapse "
            "into a single Librarian invocation per window."
        ),
    )
    p.add_argument(
        "--session-end-debounce-secs",
        type=float,
        default=DEFAULT_SESSION_END_DEBOUNCE_SECS,
        help=(
            f"Shorter debounce window applied on SessionEnd (default: "
            f"{DEFAULT_SESSION_END_DEBOUNCE_SECS}s) — the session is over, "
            "don't make it wait the full Stop window."
        ),
    )
    p.add_argument(
        "--scholar-every-k",
        type=int,
        default=DEFAULT_SCHOLAR_EVERY_K,
        help=f"Run Scholar after K packets queue up (default: {DEFAULT_SCHOLAR_EVERY_K}).",
    )
    p.add_argument(
        "--scholar-every-m-secs",
        type=float,
        default=DEFAULT_SCHOLAR_EVERY_M_SECS,
        help=f"...or every M seconds, whichever first (default: {DEFAULT_SCHOLAR_EVERY_M_SECS}).",
    )
    p.add_argument(
        "--scholar-max-batch",
        type=int,
        default=DEFAULT_SCHOLAR_MAX_BATCH,
        help=(
            f"Cap on packets per Scholar batch (default: "
            f"{DEFAULT_SCHOLAR_MAX_BATCH}). Stops a runaway burst from "
            "producing a giant prompt."
        ),
    )
    p.add_argument(
        "--queue-ceiling",
        type=int,
        default=DEFAULT_QUEUE_CEILING,
        help=(
            f"Bounded queue capacity for both the Librarian-input and "
            f"Scholar-input queues (default: {DEFAULT_QUEUE_CEILING}). When "
            "full, new items drop with a WARN."
        ),
    )
    p.add_argument(
        "--sweep-interval-secs",
        type=float,
        default=DEFAULT_SWEEP_INTERVAL_SECS,
        help=f"Buffer sweep cadence (default: {DEFAULT_SWEEP_INTERVAL_SECS}s).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG-level logging.",
    )
    return p


# ---- main loop ------------------------------------------------------


def _prewarm_indexes(knowledge_root: Path, log: logging.Logger) -> None:
    """Verify the BM25 and embedding indexes match the markdown source
    of truth at startup. Rebuild on drift; log either way.

    Run synchronously before worker threads start so first retrieval
    doesn't pay the rebuild cost. Failures fail-soft to lazy rebuild —
    a broken index never blocks daemon start.

    Note: bm25 and embeddings live in a sibling ``tools/search`` package
    not always on the daemon's ``sys.path``; the imports are kept local
    to this function and wrapped in best-effort ``try/except`` so a
    missing module degrades to lazy rebuild rather than blocking startup.
    """
    if not knowledge_root.exists():
        log.info("startup: knowledge dir %s missing — skipping index prewarm", knowledge_root)
        return
    try:
        from bm25 import load_or_build as bm25_load  # noqa: PLC0415

        idx = bm25_load(knowledge_root)
        log.info("startup: bm25 index ready (%d docs)", len(idx.docs))
    except Exception:
        log.exception("startup: bm25 prewarm failed (lazy rebuild on first use)")
    try:
        from embeddings import load_or_build as emb_load  # noqa: PLC0415

        idx = emb_load(knowledge_root)
        log.info("startup: embedding index ready (%d docs)", len(idx.docs))
    except Exception:
        log.exception("startup: embedding prewarm failed (lazy rebuild on first use)")


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def handler(signum: int, _frame: FrameType | None) -> None:
        # signal.Signals(signum).name avoids hardcoding numbers in the
        # log line — clearer when reading post-mortem.
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        logging.getLogger("agent_mem_daemon").info("received %s; shutting down", name)
        stop_event.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def run(args: argparse.Namespace) -> int:
    ensure_home()
    events_file = args.events_file or events_path()
    logfile = args.log_file or log_path()
    pidfile = args.pid_file or pid_path()

    log = configure_logging(
        logfile,
        level=logging.DEBUG if args.verbose else logging.INFO,
        foreground=args.foreground,
    )
    log.info("agent-mem-daemon starting (pid=%d)", os.getpid())
    log.info("  events file: %s", events_file)
    log.info("  log file:    %s", logfile)
    log.info("  pid file:    %s", pidfile)

    acquire_pid_file(pidfile)

    # Verify indexes are in sync with the markdown source of truth.
    # Rebuilds on drift (manual edits, git pull from another machine,
    # restore from backup); logs "ready (N docs)" otherwise. Failures
    # fail-soft — first retrieval rebuilds.
    _prewarm_indexes(knowledge_dir(), log)

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    buf = RollingBuffer(
        max_turns=args.max_turns,
        inactivity_seconds=args.inactivity_seconds,
    )
    sched = Scheduler(
        buffer=buf,
        config=SchedulerConfig(
            librarian_concurrency=args.librarian_concurrency,
            librarian_debounce_secs=args.librarian_debounce_secs,
            session_end_debounce_secs=args.session_end_debounce_secs,
            scholar_every_k=args.scholar_every_k,
            scholar_every_m_secs=args.scholar_every_m_secs,
            scholar_max_batch=args.scholar_max_batch,
            queue_ceiling=args.queue_ceiling,
            sweep_interval_secs=args.sweep_interval_secs,
        ),
    )
    tailer = JsonlTailer(
        events_file,
        sched.on_event,
        poll_interval=args.poll_interval,
        stop_event=stop_event,
        offset_state_path=offset_state_path(),
    )

    # v2: the tailer runs on its own thread, the scheduler owns a
    # debounce thread + librarian pool + scholar worker. The main
    # thread just waits for shutdown — all the work happens in the
    # background.
    log.info("entering main loop (threaded worker model)")
    tailer_thread = TailerThread(
        tailer=tailer,
        poll_interval=args.poll_interval,
        stop_event=stop_event,
    )
    # Tier-1 priming RPC: the UserPromptSubmit hook connects here per
    # turn and gets a freshly-rendered priming snippet (keyed on the
    # user's prompt, not the Librarian's batch). Lives on its own
    # daemon thread with its own stop_event so a separate ctrl-flow
    # bug in the scheduler can't leave the socket bound.
    rpc_thread = PrimingRpcThread(stop_event=threading.Event())
    sched.start()
    tailer_thread.start()
    rpc_thread.start()
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        log.info("main loop exit: signalling threads to drain")
        # Stop the tailer first so no new events arrive while the
        # debouncer is shutting down.
        stop_event.set()
        tailer_thread.join(timeout=5.0)
        # Stop the scheduler: drains in-flight librarian work,
        # then drains the scholar queue with one final batch.
        sched.stop()
        # Close the priming socket and unlink the socket file so a
        # crashed daemon never leaves a stale path that blocks the
        # next start. Last — the hook can keep getting priming until
        # the moment we shut down.
        rpc_thread.stop(timeout=2.0)
        log.info(
            "shutdown complete: stats=librarian_runs=%d backpressure_skips=%d "
            "librarian_debounced=%d scholar_runs=%d packets_queued=%d "
            "packets_drained=%d high_water=%d scholar_backpressure_skips=%d",
            sched.stats.librarian_runs,
            sched.stats.librarian_skipped_backpressure,
            sched.stats.librarian_debounced,
            sched.stats.scholar_runs,
            sched.stats.packets_queued_total,
            sched.stats.packets_drained_total,
            sched.stats.queue_high_water,
            sched.stats.scholar_skipped_backpressure,
        )
        release_pid_file(pidfile)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # TODO(phase-4): if not args.foreground, double-fork or hand off to
    # launchd here. For now there is no background mode.
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
