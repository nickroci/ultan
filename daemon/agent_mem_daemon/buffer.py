"""Per-session rolling buffer of recent turns.

PLAN §1 component: "Maintains a rolling buffer of the last N turns per
session (default N=20)."

Semantics (PLAN §4 + scope brief):
- A "turn" is bounded by ``Stop`` events. PostToolUse events between two
  Stops belong to the same turn.
- We don't see user prompts in the event stream as a distinct event type
  in the skeleton — the hook author may add UserPromptSubmit later. For
  now any event whose ``type`` isn't ``Stop`` accumulates into the
  open turn.
- On a ``Stop`` event, the open turn is sealed and pushed onto the
  session's deque (capped at N).
- ``SessionEnd`` also seals the open turn (if non-empty) and marks the
  session as ended. We keep it around until the inactivity sweep evicts
  it, because the Librarian may still want to consume the final state.
- Sessions idle for ``inactivity_seconds`` (default 1h) are dropped.

Thread-safety: the v2 scheduler runs the tailer, the debounce timer
thread, and a librarian worker pool concurrently. All mutating /
reading methods on :class:`RollingBuffer` acquire an internal
:class:`threading.RLock` so multiple threads can safely call
``ingest``, ``snapshot``, ``sweep`` etc. without external locking.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


DEFAULT_MAX_TURNS = 20
DEFAULT_INACTIVITY_SECONDS = 60 * 60  # 1 hour


@dataclass
class Event:
    """One parsed line from the events JSONL."""

    ts: float                # unix seconds; falls back to receipt time
    session_id: str
    type: str                # PostToolUse | Stop | SessionEnd | ...
    cwd: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """One sealed turn (a list of events between two Stops)."""

    events: List[Event] = field(default_factory=list)
    started_at: float = 0.0
    sealed_at: float = 0.0

    def is_empty(self) -> bool:
        return not self.events


@dataclass
class SessionState:
    session_id: str
    cwd: Optional[str] = None
    turns: Deque[Turn] = field(default_factory=deque)
    open_turn: Turn = field(default_factory=Turn)
    # Set to the ts of the most recent event ingested for this session.
    # We trust the event's ts as the source of truth (it's the hook's
    # timestamp); tests rely on being able to backdate events to
    # exercise the sweep. Initialised to 0.0 so a session created with
    # no events yet is eligible for eviction immediately — which
    # never happens in practice (sessions are created by ingest()).
    last_activity: float = 0.0
    ended: bool = False
    # Set by the scheduler when a Stop arrives; cleared when the
    # Librarian has consumed it. Lives here because it is per-session
    # state that survives across scheduler ticks.
    needs_librarian: bool = False


class RollingBuffer:
    """All live sessions in memory.

    Thread-safe: every public method acquires an internal
    :class:`threading.RLock` before touching ``self._sessions`` or the
    per-session state. Reentrant because :meth:`ingest` calls
    :meth:`_seal_turn` while already holding the lock.
    """

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        inactivity_seconds: float = DEFAULT_INACTIVITY_SECONDS,
    ) -> None:
        self.max_turns = max_turns
        self.inactivity_seconds = inactivity_seconds
        self._sessions: Dict[str, SessionState] = {}
        # Reentrant so ingest() -> _seal_turn() doesn't deadlock and so
        # callers can compose snapshot()+sweep() under a single lock if
        # they ever need to. All public methods take this.
        self._lock = threading.RLock()

    # ---- ingestion --------------------------------------------------

    def ingest(self, ev: Event) -> Optional[SessionState]:
        """Fold one event into its session. Returns the session if a
        ``Stop`` sealed a turn (so the scheduler can mark it needing a
        Librarian pass); otherwise None.
        """
        with self._lock:
            sess = self._sessions.get(ev.session_id)
            if sess is None:
                sess = SessionState(session_id=ev.session_id, cwd=ev.cwd)
                self._sessions[ev.session_id] = sess

            # cwd may arrive late (first event lacks it). Keep the first
            # non-empty value seen.
            if ev.cwd and not sess.cwd:
                sess.cwd = ev.cwd

            sess.last_activity = ev.ts

            if ev.type == "Stop":
                return self._seal_turn(sess, ev)

            if ev.type == "SessionEnd":
                self._seal_turn(sess, ev)
                sess.ended = True
                sess.needs_librarian = True
                return sess

            # Anything else (PostToolUse, UserPromptSubmit, custom) goes
            # into the open turn.
            if sess.open_turn.is_empty():
                sess.open_turn.started_at = ev.ts
            sess.open_turn.events.append(ev)
            return None

    def _seal_turn(self, sess: SessionState, sealing_ev: Event) -> SessionState:
        # The Stop event itself is also part of the turn — keeps the
        # event stream lossless when a future Librarian wants to read it.
        # Caller already holds ``self._lock``.
        if sess.open_turn.is_empty():
            sess.open_turn.started_at = sealing_ev.ts
        sess.open_turn.events.append(sealing_ev)
        sess.open_turn.sealed_at = sealing_ev.ts
        sess.turns.append(sess.open_turn)
        while len(sess.turns) > self.max_turns:
            sess.turns.popleft()
        sess.open_turn = Turn()
        sess.needs_librarian = True
        return sess

    # ---- maintenance ------------------------------------------------

    def sweep(self, *, now: Optional[float] = None) -> List[str]:
        """Drop sessions idle for > ``inactivity_seconds``.

        Returns the list of dropped session ids (mostly for logging /
        testing).
        """
        if now is None:
            now = time.time()
        with self._lock:
            dropped: List[str] = []
            cutoff = now - self.inactivity_seconds
            for sid, sess in list(self._sessions.items()):
                if sess.last_activity < cutoff:
                    del self._sessions[sid]
                    dropped.append(sid)
            return dropped

    # ---- accessors --------------------------------------------------

    def session(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def sessions(self) -> List[SessionState]:
        with self._lock:
            return list(self._sessions.values())

    def snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Materialise a session's recent turns as a plain dict for
        handoff to the Librarian. We deliberately do not pass the
        ``SessionState`` object — the future Librarian implementation
        will be in another module and should not couple to our internals.

        Returns a fully-detached copy of the session state so the caller
        can hand it to a worker thread without any further locking.
        """
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return None
            return {
                "session_id": sess.session_id,
                "cwd": sess.cwd,
                "ended": sess.ended,
                "last_activity": sess.last_activity,
                "turns": [
                    {
                        "started_at": t.started_at,
                        "sealed_at": t.sealed_at,
                        "events": [
                            {
                                "ts": e.ts,
                                "type": e.type,
                                "cwd": e.cwd,
                                "payload": e.payload,
                            }
                            for e in t.events
                        ],
                    }
                    for t in sess.turns
                ],
            }
