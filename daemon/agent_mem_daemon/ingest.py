"""JSONL-tail event ingester.

PLAN §7 open question #3 leans JSONL-tail over Unix-socket because it
survives daemon restarts (events buffer on disk). This module is the
read side of that contract.

Robustness requirements:

1. **Rotation / truncation.** The hook author may rotate the file (mv +
   recreate) or truncate it. We detect both: each poll re-stats the path
   and compares ``(st_ino, st_dev)`` against the tuple we opened. If
   either changes — or if the file shrank below our offset — we close
   the current handle and re-open from the top.

2. **Partial lines.** A hook may have flushed half a line when we read.
   We buffer the trailing fragment and prepend it to the next read.

3. **Bad JSON.** Logged and skipped — never crashes the loop. The whole
   point of JSONL-tail is robustness; one malformed line shouldn't take
   the daemon down.

4. **Cooperative shutdown.** ``run_forever`` checks ``stop_event``
   between polls so SIGTERM/SIGINT can drain in one poll interval.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TextIO

from .buffer import Event

log = logging.getLogger("agent_mem_daemon.ingest")

# How long to wait when there's nothing new on the file. Short enough
# that Stop-event handoff feels near-real-time; long enough not to spin.
DEFAULT_POLL_INTERVAL = 0.25


# Recognised event types. The ingester accepts anything but logs at
# DEBUG when it sees a type it doesn't recognise — this is forward-
# compatibility for the hook author adding UserPromptSubmit etc.
KNOWN_TYPES = {
    "PostToolUse",
    "PreToolUse",
    "Stop",
    "SessionEnd",
    "UserPromptSubmit",
    "SessionStart",
}


@dataclass
class _OpenFile:
    """The file we're currently tailing, plus identity for rotation
    detection."""

    path: Path
    fh: TextIO  # text-mode file handle from open(path, "r", encoding="utf-8")
    inode: int
    dev: int
    offset: int = 0
    # Size and mtime as of the last poll. Letting us check both means
    # we catch the pathological case where the file is truncated and
    # re-written to roughly the same size between two polls — size
    # alone won't show it, but mtime will. Per scope brief: "mtime +
    # inode tracking."
    last_size: int = 0
    last_mtime_ns: int = 0
    pending: str = ""  # partial-line buffer


def _stat_identity(path: Path) -> Optional[tuple]:
    try:
        s = path.stat()
    except FileNotFoundError:
        return None
    return (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns)


def _load_offset_state(state_path: Path) -> Optional[dict]:
    """Read the persisted offset state. Returns None if missing/corrupt."""
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _save_offset_state(state_path: Path, of: "_OpenFile") -> None:
    """Persist the tailer's current position. Atomic via tmp+rename."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        payload = {
            "path": str(of.path),
            "inode": of.inode,
            "dev": of.dev,
            "offset": of.offset,
            "last_size": of.last_size,
            "last_mtime_ns": of.last_mtime_ns,
        }
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, state_path)
    except OSError as e:
        log.debug("could not persist offset state to %s: %s", state_path, e)


def _open(
    path: Path,
    *,
    start_from_end: bool = False,
    offset_state_path: Optional[Path] = None,
) -> Optional[_OpenFile]:
    """Open ``path`` for tailing.

    Starting offset is determined by:
      1. If ``offset_state_path`` is given and contains a valid resume
         point (matching inode/dev, offset within current file bounds),
         seek to that offset. This is the production path — it lets the
         daemon restart cleanly without missing events that landed
         while it was down.
      2. Otherwise, if ``start_from_end`` is True, seek to EOF. This is
         legacy behaviour kept for tests that explicitly opt in.
      3. Otherwise (the default), seek to 0. We read everything the
         file currently holds, then continue from there. This handles
         the cold-start race where events were written to a newly
         created events.jsonl *just before* the daemon's first poll
         attached — those events would otherwise be lost to the
         seek-to-EOF behaviour.
    """
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError as e:
        log.warning("could not open events file %s: %s", path, e)
        return None
    try:
        st = os.fstat(fh.fileno())
    except OSError:
        fh.close()
        return None

    # 1. Try persisted offset.
    resume_offset: Optional[int] = None
    if offset_state_path is not None:
        state = _load_offset_state(offset_state_path)
        if (
            state is not None
            and state.get("inode") == st.st_ino
            and state.get("dev") == st.st_dev
            and isinstance(state.get("offset"), int)
            and 0 <= state["offset"] <= st.st_size
        ):
            resume_offset = state["offset"]

    if resume_offset is not None:
        fh.seek(resume_offset)
        log.info(
            "tailer: resuming %s at offset %d (file size %d)",
            path,
            resume_offset,
            st.st_size,
        )
    elif start_from_end:
        fh.seek(0, os.SEEK_END)
    else:
        fh.seek(0)
        if st.st_size > 0:
            log.info(
                "tailer: opening %s from start (size %d) — no resume state",
                path,
                st.st_size,
            )

    return _OpenFile(
        path=path,
        fh=fh,
        inode=st.st_ino,
        dev=st.st_dev,
        offset=fh.tell(),
        last_size=st.st_size,
        last_mtime_ns=st.st_mtime_ns,
    )


def _reopen_from_start(of: _OpenFile) -> Optional[_OpenFile]:
    """File was rotated/truncated — close + re-open from offset 0."""
    try:
        of.fh.close()
    except OSError:
        pass
    try:
        fh = open(of.path, "r", encoding="utf-8", errors="replace")
        st = os.fstat(fh.fileno())
    except (FileNotFoundError, OSError) as e:
        log.debug("re-open of %s failed: %s", of.path, e)
        return None
    return _OpenFile(
        path=of.path,
        fh=fh,
        inode=st.st_ino,
        dev=st.st_dev,
        offset=0,
        last_size=st.st_size,
        last_mtime_ns=st.st_mtime_ns,
        pending="",
    )


def parse_event_line(line: str, *, now_fn: Callable[[], float] = time.time) -> Optional[Event]:
    """Parse one JSONL line into an Event. Returns None on bad input.

    Schema (the contract with the hook author):
        {
          "ts": "2026-05-19T10:30:00Z"  | float,    # required, ISO-8601 or unix
          "session_id": "abc-123",                  # required
          "type": "PostToolUse" | "Stop" | "SessionEnd" | ...,  # required
          "cwd": "/path/to/project",                # optional
          "payload": { ... arbitrary ... }          # optional
        }

    Missing required fields => None + warning. The ``ts`` field accepts
    either a numeric unix timestamp or an ISO-8601 string; missing ts
    falls back to receipt time so a sloppy hook still produces usable
    events (we log a warning).
    """
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        log.warning("bad JSON in events line: %s (line=%r)", e, line[:200])
        return None
    if not isinstance(obj, dict):
        log.warning("event line is not an object: %r", line[:200])
        return None

    sid = obj.get("session_id")
    typ = obj.get("type")
    if not sid or not typ:
        log.warning("event missing session_id/type: %r", obj)
        return None

    ts_raw = obj.get("ts")
    ts = _coerce_ts(ts_raw)
    if ts is None:
        log.warning("event has no parseable ts, using receipt time: %r", obj)
        ts = now_fn()

    if typ not in KNOWN_TYPES:
        log.debug("unknown event type %r — ingesting anyway", typ)

    return Event(
        ts=ts,
        session_id=str(sid),
        type=str(typ),
        cwd=obj.get("cwd"),
        payload=obj.get("payload") or {},
        raw=obj,
    )


def _coerce_ts(ts_raw) -> Optional[float]:
    """Accept unix-seconds (int/float) or ISO-8601. Returns float or None."""
    if ts_raw is None:
        return None
    if isinstance(ts_raw, (int, float)):
        return float(ts_raw)
    if isinstance(ts_raw, str):
        # Try unix-seconds-as-string first; cheap path.
        try:
            return float(ts_raw)
        except ValueError:
            pass
        # ISO-8601. Python's fromisoformat is strict about 'Z'; rewrite
        # to '+00:00' before parsing.
        try:
            from datetime import datetime

            s = ts_raw.replace("Z", "+00:00")
            return datetime.fromisoformat(s).timestamp()
        except (ValueError, ImportError):
            return None
    return None


class JsonlTailer:
    """Polls a JSONL file and emits parsed Events through a callback.

    Single-threaded by default; ``run_forever`` blocks. Tests can call
    ``poll_once`` directly.
    """

    def __init__(
        self,
        path: Path,
        on_event: Callable[[Event], None],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        stop_event: Optional[threading.Event] = None,
        start_from_end: bool = False,
        offset_state_path: Optional[Path] = None,
    ) -> None:
        self.path = path
        self.on_event = on_event
        self.poll_interval = poll_interval
        self.stop_event = stop_event or threading.Event()
        self._of: Optional[_OpenFile] = None
        self._start_from_end = start_from_end
        self._offset_state_path = offset_state_path

    def _attach(self) -> None:
        """Open the file if it exists. Resume from persisted offset when
        available; otherwise read from start. ``start_from_end=True``
        forces seek-to-EOF (legacy / test-only)."""
        self._of = _open(
            self.path,
            start_from_end=self._start_from_end,
            offset_state_path=self._offset_state_path,
        )

    def _persist(self) -> None:
        if self._of is not None and self._offset_state_path is not None:
            _save_offset_state(self._offset_state_path, self._of)

    def poll_once(self) -> int:
        """Drain whatever's been appended since the last poll. Returns
        the number of events emitted (useful for tests).

        Handles three transitions:
          - file did not exist; now does => attach + read
          - file was rotated (new inode) => re-open from start
          - file was truncated (size < offset) => re-open from start
        """
        if self._of is None:
            self._attach()
            if self._of is None:
                return 0

        of = self._of
        ident = _stat_identity(of.path)
        if ident is None:
            # File vanished. Drop the handle; next poll will retry.
            try:
                of.fh.close()
            except OSError:
                pass
            self._of = None
            return 0

        inode, dev, size, mtime_ns = ident
        if inode != of.inode or dev != of.dev:
            log.info("events file rotated (inode %s -> %s); re-opening", of.inode, inode)
            self._of = _reopen_from_start(of)
            if self._of is None:
                return 0
            of = self._of
        elif size < of.offset or size < of.last_size:
            # Truncation: current size is below where we last read to,
            # OR below the size we observed on the previous poll
            # (catches grow-then-shrink between polls).
            log.info(
                "events file truncated (size=%d, offset=%d, last_size=%d); re-opening",
                size,
                of.offset,
                of.last_size,
            )
            self._of = _reopen_from_start(of)
            if self._of is None:
                return 0
            of = self._of
        elif (
            mtime_ns != of.last_mtime_ns
            and size == of.last_size
            and of.last_size > 0
            and of.offset >= size
        ):
            # Pathological case: file was truncated and re-written to
            # the *same* size between two polls. Size comparison can't
            # see it; mtime advancing while size held constant — and
            # our read position is at or past EOF — is the giveaway.
            # We could miss this if the rewrite happens to land in the
            # same instant as the last mtime, but at filesystem-ns
            # resolution that's effectively zero.
            log.info("events file rewritten in place (mtime advanced, size unchanged); re-opening")
            self._of = _reopen_from_start(of)
            if self._of is None:
                return 0
            of = self._of

        emitted = 0
        try:
            chunk = of.fh.read()
        except OSError as e:
            log.warning("read failed on %s: %s", of.path, e)
            return 0
        if not chunk:
            # Even with no new bytes to read, update last_size +
            # last_mtime_ns so we don't falsely detect a state change
            # on the next poll.
            of.last_size = size
            of.last_mtime_ns = mtime_ns
            return 0

        of.offset = of.fh.tell()
        of.last_size = of.offset
        of.last_mtime_ns = mtime_ns
        buf = of.pending + chunk
        # Split on newline; keep the trailing partial (if any) for the
        # next poll. ``splitlines(keepends=False)`` would discard the
        # signal of whether the buffer ended on a newline, so we split
        # manually.
        if buf.endswith("\n"):
            lines = buf[:-1].split("\n")
            of.pending = ""
        else:
            lines = buf.split("\n")
            of.pending = lines.pop()  # last element is the partial

        for line in lines:
            ev = parse_event_line(line)
            if ev is None:
                continue
            try:
                self.on_event(ev)
                emitted += 1
            except Exception:
                # Callback errors must not break the tail. Log + carry on.
                log.exception("on_event callback raised")
        # Persist offset after each successful drain so a daemon restart
        # resumes here instead of seeking to EOF and losing events.
        self._persist()
        return emitted

    def run_forever(self) -> None:
        log.info("tailing events file: %s (poll=%.2fs)", self.path, self.poll_interval)
        while not self.stop_event.is_set():
            self.poll_once()
            # Use wait() not sleep() so SIGTERM drains fast.
            self.stop_event.wait(self.poll_interval)
        if self._of is not None:
            try:
                self._of.fh.close()
            except OSError:
                pass
            self._of = None
        log.info("tail stopped")
