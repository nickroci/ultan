"""Daemon-side priming RPC over a Unix-domain socket.

Why this exists
---------------
The Tier 1 (ambient priming) pipeline used to refresh ``hot-context.md``
on the back of every Scholar batch, using the Librarian's proposals as
the BM25/embedding query. That's the wrong trigger and the wrong
question:

  - **Wrong trigger.** A normal Librarian batch emits 0 proposals most
    of the time, which collapses the query to empty and clears the file.
    The agent then gets nothing primed.
  - **Wrong question.** Even when the Librarian DID emit proposals, the
    file the hook injects on the *next* turn is keyed on the *previous*
    batch's curation summary, not on what the user just asked.

We want priming to be answered against the user's prompt at the moment
they submit it. That has to happen inside the ``UserPromptSubmit`` hook
— but the hook is a fresh Python process per invocation, and loading
the sentence-transformer model in-process is ~5s. Too slow.

Solution: the always-running daemon (model warm in memory) exposes a
Unix-domain socket. The hook becomes a thin socket client that ships
the user's prompt over and gets rendered markdown back. The daemon
reuses ``priming._hybrid_search`` + ``priming._boost_with_reinforcement``
+ ``priming._assemble_output`` so the output is byte-for-byte the same
shape as the old ``hot-context.md``.

Protocol (kept simple — no gRPC, no msgpack, just stdlib):

  - Length-prefixed JSON over a SOCK_STREAM Unix socket.
  - 4 bytes big-endian length, then that many bytes of UTF-8 JSON.
  - Request: ``{"op": "priming", "prompt": str, "project_slug": str|None,
    "k": int, "char_budget": int}``.
  - Success response: ``{"ok": true, "priming_md": str, "took_ms": int,
    "lane": "hybrid"|"bm25"}``.
  - Failure response: ``{"ok": false, "error": str}``.

Lifecycle: one accept thread (daemon=True). Each accepted connection is
handed to a small ThreadPoolExecutor — request volume is low (one per
user prompt, typically << 1/sec), so the executor size is small. The
server caps each request at ``SERVER_REQUEST_TIMEOUT_S`` to keep a
stuck handler from hogging a worker.

The handler never raises out to the accept loop; every exception lands
in the JSON failure response and the connection closes cleanly. The
embedding model load happens once at first call (lazy import inside
``priming._hybrid_search``) — module-level caches in ``embeddings.py``
mean subsequent requests are warm.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, Optional, cast

from bm25 import load_or_build as bm25_load_or_build

from . import decay, priming
from .paths import knowledge_dir, priming_socket_path

# Wire-level type aliases. JSON parses into arbitrary nested values, so
# we type incoming requests as Mapping[str, object] and use isinstance
# narrowing at each access site. Responses are dicts the dispatch layer
# serialises via json.dumps, so dict[str, object] is the right shape.
RpcRequest = Mapping[str, object]
RpcResponse = dict[str, object]
RpcHandler = Callable[[RpcRequest], RpcResponse]

log = logging.getLogger("agent_mem_daemon.priming_rpc")


# ── Tunables ─────────────────────────────────────────────────────────


# Hard cap on a single request handler's wall time. Sits below the hook
# client's 2s total budget so a slow handler still surfaces as a server
# error rather than a client-side socket timeout (cleaner failure mode).
# The cross-encoder rerank stage runs ~300ms warm on Apple Silicon; CPU
# or older GPUs can push that to ~1s, hence the headroom here.
SERVER_REQUEST_TIMEOUT_S = 1.8

# Length-prefix size (big-endian uint32). 4 GB ceiling per message — we
# expect bodies in the kilobyte range; the limit exists to refuse
# obvious garbage / malicious clients without a separate validator.
_LEN_HEADER = 4
_MAX_BODY_BYTES = 1 << 20  # 1 MiB; priming markdown is ~500 chars

# Concurrent in-flight requests. Each handler is short-lived (≤1s by
# cap) and mostly I/O-bound, so a small pool is plenty.
_HANDLER_POOL_SIZE = 4

# Per-session sent-wikilink cache so the agent doesn't see the same
# priming bullets surfaced over and over within one session. Bounded
# LRU on the (session_id) key; each value is a bounded set of wikilinks
# already shown. Numbers chosen to be tiny — even at the cap, total
# memory is well under 1 MB and well within hook-budget lookup cost.
_SENT_CACHE_MAX_SESSIONS = 64
_SENT_CACHE_MAX_LINKS_PER_SESSION = 200

_sent_cache_lock = threading.Lock()
# OrderedDict[session_id, OrderedDict[wikilink, None]]. Inner dict uses
# OrderedDict-as-ordered-set so we can LRU-evict the oldest links per
# session without pulling in a separate set + deque. Both layers need
# explicit type parameters so pyright doesn't widen them to Unknown.
_sent_cache: "OrderedDict[str, OrderedDict[str, None]]" = OrderedDict()


def _sent_get(session_id: str) -> set[str]:
    """Return the set of wikilinks already shown to ``session_id``.

    Promotes ``session_id`` to MRU position on access. Returns a copy
    so callers can iterate / pass to ``_assemble_output`` without
    holding the lock.
    """
    with _sent_cache_lock:
        links: Optional[OrderedDict[str, None]] = _sent_cache.get(session_id)
        if links is None:
            return set()
        _sent_cache.move_to_end(session_id)
        return set(links.keys())


def _sent_record(session_id: str, new_links: list[str]) -> None:
    """Mark ``new_links`` as sent to ``session_id``; evict old entries
    when over either cap. Idempotent on repeated links."""
    if not new_links:
        return
    with _sent_cache_lock:
        bucket: Optional[OrderedDict[str, None]] = _sent_cache.get(session_id)
        if bucket is None:
            bucket = OrderedDict()
            _sent_cache[session_id] = bucket
        for link in new_links:
            if link in bucket:
                bucket.move_to_end(link)
            else:
                bucket[link] = None
                while len(bucket) > _SENT_CACHE_MAX_LINKS_PER_SESSION:
                    bucket.popitem(last=False)
        _sent_cache.move_to_end(session_id)
        while len(_sent_cache) > _SENT_CACHE_MAX_SESSIONS:
            _sent_cache.popitem(last=False)


# ── Wire helpers ─────────────────────────────────────────────────────


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes or return ``None`` on EOF.

    Returns ``None`` (rather than raising) so the handler can treat a
    half-open connection as "client gave up" and log a single line
    instead of an exception traceback per dropped connection.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (ConnectionResetError, BrokenPipeError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _send_message(sock: socket.socket, body: bytes) -> None:
    """Length-prefixed write. Raises on socket error so caller logs."""
    header = struct.pack(">I", len(body))
    sock.sendall(header + body)


def _recv_message(sock: socket.socket) -> Optional[bytes]:
    """Read one length-prefixed message; ``None`` on EOF or oversize."""
    header = _recv_exact(sock, _LEN_HEADER)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > _MAX_BODY_BYTES:
        return None
    return _recv_exact(sock, length)


# ── Request handler ──────────────────────────────────────────────────


def _handle_priming(req: RpcRequest) -> RpcResponse:
    """Render a priming snippet for ``req['prompt']``.

    Reuses ``priming._hybrid_search`` (BM25 + embeddings via RRF) and
    ``priming._assemble_output`` so the rendered markdown is identical
    to the old ``hot-context.md`` shape — drop-in replacement on the
    hook side.

    Returns a wire-ready dict (no socket I/O here).
    """
    raw_prompt: object = req.get("prompt") or ""
    if not isinstance(raw_prompt, str):
        return {"ok": False, "error": "prompt must be a string"}
    prompt = raw_prompt.strip()
    if not prompt:
        # Empty prompt -> empty render, but not an error. The hook
        # treats empty markdown as "nothing to inject" and moves on.
        return {"ok": True, "priming_md": "", "took_ms": 0, "lane": "bm25"}

    raw_k: object = req.get("k", 3)
    raw_budget: object = req.get("char_budget", 1500)
    if not isinstance(raw_k, (int, float, str)) or not isinstance(raw_budget, (int, float, str)):
        return {"ok": False, "error": "k and char_budget must be ints"}
    try:
        k = int(raw_k)
        char_budget = int(raw_budget)
    except (TypeError, ValueError):
        return {"ok": False, "error": "k and char_budget must be ints"}
    if k <= 0 or char_budget <= 0:
        return {"ok": False, "error": "k and char_budget must be positive"}

    raw_slug: object = req.get("project_slug")
    current_project_slug: Optional[str] = (
        raw_slug.strip() if isinstance(raw_slug, str) and raw_slug.strip() else None
    )

    raw_session: object = req.get("session_id")
    session_id: Optional[str] = (
        raw_session.strip() if isinstance(raw_session, str) and raw_session.strip() else None
    )
    # Dedup is per-session. Without a session_id we can't track state,
    # so the rendering path runs with an empty already_sent set — the
    # agent gets full priming as if it were a fresh session.
    already_sent: set[str] = _sent_get(session_id) if session_id else set()

    kdir = knowledge_dir()
    if not kdir.exists():
        return {"ok": True, "priming_md": "", "took_ms": 0, "lane": "bm25"}

    t0 = time.monotonic()
    # Pull a few extras so the reinforcement boost can promote a
    # low-BM25-but-heavily-reinforced entry into the top_k. Mirrors
    # priming.refresh_hot_context's strategy.
    hits = priming._hybrid_search(  # pyright: ignore[reportPrivateUsage]  # sibling-module helper
        kdir,
        prompt,
        k=max(k * 2, k + 3),
    )
    # Best-effort lane detection: if embeddings are unavailable the
    # embedding lane returns []; we report bm25 in that case so the
    # operator can see at a glance whether the model is loaded.
    lane = "hybrid"
    try:
        __import__("embeddings")
    except Exception:
        lane = "bm25"

    if not hits:
        took_ms = int((time.monotonic() - t0) * 1000)
        return {"ok": True, "priming_md": "", "took_ms": took_ms, "lane": lane}

    ranked = priming._boost_with_reinforcement(  # pyright: ignore[reportPrivateUsage]
        hits,
        knowledge_dir=kdir,
        current_project_slug=current_project_slug,
    )
    body, newly_sent = priming._assemble_output(  # pyright: ignore[reportPrivateUsage]
        ranked,
        kdir,
        top_k=k,
        char_budget=char_budget,
        already_sent=already_sent,
    )
    _post_render_bookkeeping(kdir, session_id, body, newly_sent)

    took_ms = int((time.monotonic() - t0) * 1000)
    return {"ok": True, "priming_md": body or "", "took_ms": took_ms, "lane": lane}


def _post_render_bookkeeping(
    kdir: Path,
    session_id: Optional[str],
    body: str,
    newly_sent: list[str],
) -> None:
    """Three side-effects after a successful priming render:

    1. Per-session sent-cache update (only when body non-empty and
       session_id present — otherwise there's no state to track).
    2. Surface bookkeeping on each newly-shown entry via
       ``decay.record_surface``: increment the ``fired`` counter and
       stamp ``last_surfaced``, so the decay sweep treats surfacing as
       activity and ``fired`` tracks the surface denominator for
       ``fired-helpful``. Runs even without ``session_id`` because the
       agent saw the entries regardless of how the daemon tracks
       sessions.
    3. Opportunistic ``maybe_run_sweep`` — self-skips unless the 24h
       cooldown has elapsed. One ``stat()`` on the sweep-state file
       on the common path.

    All three are best-effort: failures are logged and swallowed so
    bookkeeping can never break the priming response.
    """
    if session_id and body and newly_sent:
        _sent_record(session_id, newly_sent)
    for link in newly_sent or ():
        entry_path = (kdir / f"{link}.md").resolve()
        try:
            decay.record_surface(entry_path)
        except Exception:
            log.exception("priming_rpc: surface bookkeeping raised for %s", link)
    try:
        decay.maybe_run_sweep(kdir)
    except Exception:
        log.exception("priming_rpc: maybe_run_sweep raised")


def _handle_bm25_search(req: RpcRequest) -> RpcResponse:
    """BM25 keyword search over the knowledge library.

    Mirrors what the in-process Librarian MCP tool offers
    (``library_tools.bm25_search``), but exposed over the daemon
    socket so end-user skills can reach it without spawning an SDK
    call. Returns a list of ``{path, score, snippet}`` hits.
    """
    query: object = req.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "query must be a non-empty string"}
    raw_k: object = req.get("k", 5)
    if not isinstance(raw_k, (int, float, str)):
        return {"ok": False, "error": "k must be an int"}
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        return {"ok": False, "error": "k must be an int"}
    k = max(1, min(20, k))

    kdir = knowledge_dir()
    if not kdir.exists():
        return {"ok": True, "hits": []}

    try:
        index = bm25_load_or_build(kdir)
    except FileNotFoundError:
        return {"ok": True, "hits": []}
    except Exception as e:
        log.exception("bm25 index load failed")
        return {"ok": False, "error": f"bm25 index error: {e}"}

    raw = index.search(query.strip(), k=k)
    kroot = kdir.resolve()
    hits: list[dict[str, object]] = []
    for p, score, snippet in raw:
        pp = Path(p)
        rel = str(pp.relative_to(kroot)) if pp.is_absolute() else str(pp)
        hits.append({"path": rel, "score": float(score), "snippet": snippet})
    return {"ok": True, "hits": hits}


def _handle_fetch_entry(req: RpcRequest) -> RpcResponse:
    """Fetch an entry + its directory context (siblings, subdirs, parent README).

    Accepts a wikilink (``[[...]]``) or plain relative path (with or
    without ``.md``). The path is resolved under the knowledge dir;
    anything escaping it is rejected. Returns the file content plus
    enough structural context for the caller to traverse without a
    second tool call per neighbour.
    """
    path_arg: object = req.get("path")
    if not isinstance(path_arg, str) or not path_arg.strip():
        return {"ok": False, "error": "path must be a non-empty string"}

    raw = path_arg.strip()
    if raw.startswith("[[") and raw.endswith("]]"):
        raw = raw[2:-2].strip()
    raw = raw.strip("/")
    if not raw:
        return {"ok": False, "error": "path must name an entry or folder"}

    kdir = knowledge_dir()
    if not kdir.exists():
        return {"ok": False, "error": "knowledge dir not found"}
    kroot = kdir.resolve()

    # Resolve to a concrete .md file. A bare folder path (with or without a
    # trailing slash) resolves to that folder's README.md — so `projects/foo`
    # and `projects/foo/` browse the folder instead of 404-ing on the
    # non-existent `projects/foo.md`. An explicit `.md` suffix is always an
    # entry; otherwise try the entry first, then fall back to the folder README.
    rels = [raw] if raw.endswith(".md") else [f"{raw}.md", f"{raw}/README.md"]
    target: Optional[Path] = None
    for rel in rels:
        cand = (kroot / rel).resolve()
        try:
            cand.relative_to(kroot)
        except ValueError:
            return {"ok": False, "error": "path escapes knowledge dir"}
        if cand.is_file():
            target = cand
            break
    if target is None:
        return {
            "ok": False,
            "error": (
                f"entry not found: {raw} (tried {', '.join(rels)}). "
                "If you meant a topic rather than a path, re-run as a free-text query."
            ),
        }

    content = target.read_text(encoding="utf-8")
    parent = target.parent
    siblings = sorted(
        f.name
        for f in parent.iterdir()
        if f.is_file() and f.suffix == ".md" and f.name != target.name
    )
    subdirs = sorted(
        f.name + "/" for f in parent.iterdir() if f.is_dir() and not f.name.startswith(".")
    )
    parent_readme = parent / "README.md"
    parent_readme_excerpt = ""
    if parent_readme.is_file() and parent_readme != target:
        text = parent_readme.read_text(encoding="utf-8")
        parent_readme_excerpt = "\n".join(text.splitlines()[:30])[:1500]

    return {
        "ok": True,
        "path": str(target.relative_to(kroot)),
        "content": content,
        "siblings": siblings,
        "subdirs": subdirs,
        "parent_readme_excerpt": parent_readme_excerpt,
    }


class RpcOp(StrEnum):
    """Wire ``op`` values the daemon RPC accepts. Adding a method = add
    an entry here and a corresponding handler to ``_HANDLERS``."""

    PRIMING = "priming"
    BM25_SEARCH = "bm25_search"
    FETCH_ENTRY = "fetch_entry"


_HANDLERS: dict[RpcOp, RpcHandler] = {
    RpcOp.PRIMING: _handle_priming,
    RpcOp.BM25_SEARCH: _handle_bm25_search,
    RpcOp.FETCH_ENTRY: _handle_fetch_entry,
}


def _dispatch(req: RpcRequest) -> RpcResponse:
    op_raw: object = req.get("op")
    if not isinstance(op_raw, str):
        return {"ok": False, "error": f"unknown op: {op_raw!r}"}
    try:
        op = RpcOp(op_raw)
    except ValueError:
        return {"ok": False, "error": f"unknown op: {op_raw!r}"}
    return _HANDLERS[op](req)


def _read_request(
    conn: socket.socket,
) -> tuple[Optional[RpcRequest], Optional[RpcResponse]]:
    """Read one length-prefixed JSON request. Returns ``(req, err_resp)``.

    Exactly one of the tuple slots is non-None:
      - ``(req, None)`` — well-formed request dict.
      - ``(None, err_resp)`` — wire-ready error response to send back.
      - ``(None, None)`` — peer closed; caller should bail without reply.
    """
    raw = _recv_message(conn)
    if raw is None:
        return None, None
    try:
        parsed: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return None, {"ok": False, "error": f"bad json: {e}"}
    if not isinstance(parsed, dict):
        return None, {"ok": False, "error": "request must be a JSON object"}
    # json.loads on an object yields dict[Any, Any]; the RPC protocol
    # only ever uses string keys, so coerce on the way in to give the
    # rest of the handler chain a fully-typed Mapping[str, object].
    parsed_dict = cast("dict[object, object]", parsed)
    req: RpcRequest = {str(k): v for k, v in parsed_dict.items()}
    return req, None


def _safe_dispatch(req: RpcRequest) -> RpcResponse:
    """Run ``_dispatch`` and turn unexpected exceptions into a JSON error."""
    try:
        return _dispatch(req)
    except Exception as e:
        log.exception("priming_rpc: handler raised")
        return {"ok": False, "error": f"handler error: {e.__class__.__name__}"}


def _serialise_response(resp: RpcResponse) -> bytes:
    """JSON-encode ``resp`` with a fallback for unserialisable payloads."""
    try:
        return json.dumps(resp).encode("utf-8")
    except (TypeError, ValueError) as e:
        return json.dumps({"ok": False, "error": f"unserializable response: {e}"}).encode("utf-8")


def _handle_connection(conn: socket.socket, addr: object) -> None:
    """One client, one request, close. Never raises.

    ``addr`` is the peer address from ``socket.accept()``; for AF_UNIX
    sockets it's typically an empty string but the contract only
    promises *something*, so we keep it as ``object`` (we don't use it).
    """
    del addr  # unused; kept positional so executor.submit's signature matches
    deadline = time.monotonic() + SERVER_REQUEST_TIMEOUT_S
    try:
        conn.settimeout(SERVER_REQUEST_TIMEOUT_S)
        req, err_resp = _read_request(conn)
        if req is None and err_resp is None:
            return
        resp: RpcResponse
        if err_resp is not None:
            resp = err_resp
        elif time.monotonic() > deadline:
            resp = {"ok": False, "error": "server-side timeout before dispatch"}
        else:
            assert req is not None  # narrowed by the branches above
            resp = _safe_dispatch(req)

        body = _serialise_response(resp)

        try:
            _send_message(conn, body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client gave up. Nothing actionable.
            pass
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass


# ── Server thread ────────────────────────────────────────────────────


class PrimingRpcThread(threading.Thread):
    """Accept loop on a Unix-domain socket. Stop via ``stop_event``.

    Cleans up the socket file on start (stale leftover from a previous
    crash) and on stop. Failures to bind are logged and the thread
    exits — the rest of the daemon continues without RPC support; the
    hook will fall back to its BM25-only path.
    """

    def __init__(
        self,
        stop_event: threading.Event,
        *,
        socket_path: Optional[Path] = None,
        pool_size: int = _HANDLER_POOL_SIZE,
    ) -> None:
        super().__init__(name="priming-rpc", daemon=True)
        self._stop_event = stop_event
        self._socket_path = socket_path or priming_socket_path()
        self._server_sock: Optional[socket.socket] = None
        self._pool = ThreadPoolExecutor(
            max_workers=pool_size, thread_name_prefix="priming-rpc-handler"
        )
        self._bound = threading.Event()

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until the server is accepting (or ``timeout`` elapses).

        Useful in tests: avoids a race between ``start()`` and the first
        client ``connect()`` while the bind is still in flight.
        """
        return self._bound.wait(timeout=timeout)

    def _bind(self) -> Optional[socket.socket]:
        # Delete any stale socket file from a previous crash. ``unlink``
        # on a non-existent path raises FileNotFoundError which we
        # swallow; any other OSError propagates to the caller's except.
        try:
            self._socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self._socket_path.exists() or self._socket_path.is_symlink():
                try:
                    self._socket_path.unlink()
                except FileNotFoundError:
                    pass
        except OSError:
            log.exception("priming_rpc: failed to clean up stale socket %s", self._socket_path)
            return None

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self._socket_path))
            sock.listen(16)
            # Owner-only — the socket file effectively gates the RPC.
            try:
                os.chmod(str(self._socket_path), 0o600)
            except OSError:
                # Best-effort; on some filesystems chmod is a no-op.
                pass
        except OSError:
            log.exception("priming_rpc: bind failed at %s", self._socket_path)
            sock.close()
            return None

        # Short accept timeout so the loop can notice stop_event in
        # reasonable time without a separate wakeup pipe.
        sock.settimeout(0.5)
        return sock

    def run(self) -> None:
        self._server_sock = self._bind()
        if self._server_sock is None:
            # Don't set _bound — wait_until_ready returns False so the
            # daemon's startup logs make the failure obvious.
            return
        self._bound.set()
        log.info("startup: priming RPC ready at %s", self._socket_path)
        try:
            while not self._stop_event.is_set():
                try:
                    conn, addr = self._server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    # Socket closed underneath us — happens during
                    # shutdown when stop() closes the server socket
                    # to break out of accept(). Exit the loop.
                    if self._stop_event.is_set():
                        break
                    log.exception("priming_rpc: accept raised")
                    continue
                try:
                    self._pool.submit(_handle_connection, conn, addr)
                except RuntimeError:
                    # Pool was shut down between the accept and submit.
                    try:
                        conn.close()
                    except OSError:
                        pass
                    break
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        # Remove the socket file so the next daemon startup doesn't
        # see a "stale" path and have to clean it up first.
        try:
            if self._socket_path.exists() or self._socket_path.is_symlink():
                self._socket_path.unlink()
        except (FileNotFoundError, OSError):
            pass

    def stop(self, timeout: float = 2.0) -> None:
        """Signal shutdown and join."""
        self._stop_event.set()
        # Closing the server socket interrupts the blocking accept;
        # the loop catches OSError and exits.
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        self.join(timeout=timeout)
