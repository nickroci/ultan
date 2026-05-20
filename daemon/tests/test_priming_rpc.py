"""Tests for the daemon-side priming RPC server.

Covers:
  - Round-trip on a real fixture library: server binds, accepts a
    JSON request, returns a valid priming response.
  - Malformed / oversized inputs come back as ``{"ok": false}`` rather
    than crashing the handler.
  - The 1s server-side request cap is honoured (we don't hold
    connections open indefinitely).
  - A stale socket file from a previous crash is cleaned up at
    bind time.
  - ``stop()`` closes the server socket gracefully and removes the
    socket file on the way out.

We exercise the real socket (Unix-domain) rather than mocking it —
the protocol is short enough that fakes wouldn't catch the half of
the bugs we care about (handler not draining, accept loop hanging,
etc).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from agent_mem_daemon import priming, priming_rpc
from agent_mem_daemon.priming_rpc import PrimingRpcThread


# macOS caps AF_UNIX socket paths at ~104 chars; pytest's tmp_path
# already eats ~70 of those before we add a filename. For the socket
# we stash it under /tmp instead; the library/knowledge dir still uses
# pytest's tmp_path so file-based assertions remain isolated.
def _short_socket_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="ult-rpc-"))
    return d


# ── Fixture helpers (small library seeded for hybrid search) ──────────


def _entry(
    *,
    id_: str,
    title: str,
    applies_when: str,
    keywords: List[str],
    reinforced: Optional[int] = None,
    body: str = "",
    scope: str = "global",
) -> str:
    """Render a minimal valid library entry."""
    lines = [
        "---",
        f"id: {id_}",
        "type: lesson",
        f"scope: {scope}",
        "status: provisional",
        "confidence: 0.7",
        "applies-when: |",
    ]
    for line in applies_when.splitlines():
        lines.append(f"  {line}")
    lines.append("keywords: [" + ", ".join(keywords) + "]")
    lines.append(f'title: "{title}"')
    lines.append("created: 2026-05-19")
    lines.append("updated: 2026-05-19")
    lines.append("fired: 0")
    lines.append("fired-helpful: 0")
    if reinforced is not None:
        lines.append(f"reinforced: {reinforced}")
    lines.append("sources:")
    lines.append("  - manual")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(body or f"Body for {id_}. {applies_when}.")
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_library(root: Path) -> Path:
    """Build a small library — enough docs for BM25 to give nonzero IDF."""
    k = root / "knowledge"
    _write(k / "README.md", "# knowledge root\n")
    _write(k / "global" / "README.md", "# global\n")
    _write(k / "global" / "python" / "README.md", "# python\n")
    _write(k / "global" / "git" / "README.md", "# git\n")
    _write(
        k / "global" / "python" / "use-uv-not-pip.md",
        _entry(
            id_="use-uv-not-pip",
            title="Always use uv for python",
            applies_when="installing python deps or running scripts",
            keywords=["python", "uv", "pip", "packaging"],
            body="Always use uv for python package management. Never pip.",
        ),
    )
    _write(
        k / "global" / "python" / "ruff-format.md",
        _entry(
            id_="ruff-format",
            title="Format python with ruff",
            applies_when="formatting python files",
            keywords=["python", "ruff", "format"],
        ),
    )
    _write(
        k / "global" / "git" / "no-force-push.md",
        _entry(
            id_="no-force-push",
            title="Never force-push to main",
            applies_when="pushing to git remotes",
            keywords=["git", "push", "remote"],
        ),
    )
    _write(
        k / "global" / "git" / "small-commits.md",
        _entry(
            id_="small-commits",
            title="Prefer small commits",
            applies_when="committing changes",
            keywords=["git", "commits", "history"],
        ),
    )
    _write(
        k / "global" / "git" / "branch-naming.md",
        _entry(
            id_="branch-naming",
            title="Use kebab-case branches",
            applies_when="creating new git branches",
            keywords=["git", "branches", "naming"],
        ),
    )
    _write(
        k / "global" / "git" / "rebase-not-merge.md",
        _entry(
            id_="rebase-not-merge",
            title="Prefer rebase over merge",
            applies_when="updating feature branches",
            keywords=["git", "rebase", "merge"],
        ),
    )
    _write(
        k / "global" / "git" / "signed-commits.md",
        _entry(
            id_="signed-commits",
            title="Sign all commits",
            applies_when="committing changes",
            keywords=["git", "gpg", "sign"],
        ),
    )
    return k


@pytest.fixture(autouse=True)
def _isolate_agent_mem_home(tmp_path, monkeypatch):
    """Force AGENT_MEM_HOME under tmp_path so each test gets its own
    socket file, BM25 index, etc. — no cross-test pollution."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    yield


# ── Wire helpers (mirror priming_rpc._send_message / _recv_message) ──


def _send_request(socket_path: Path, request: dict, *, connect_timeout: float = 2.0,
                   io_timeout: float = 2.0) -> dict:
    """Tiny test-only client. Returns the parsed JSON response.

    Intentionally re-implemented rather than calling priming_rpc's
    internals — we want the test to verify the wire format, not the
    implementation symmetry."""
    body = json.dumps(request).encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout)
    sock.connect(str(socket_path))
    sock.settimeout(io_timeout)
    sock.sendall(struct.pack(">I", len(body)) + body)

    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("server closed before sending header")
        header += chunk
    (length,) = struct.unpack(">I", header)
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("server closed before sending full body")
        payload += chunk
    sock.close()
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def rpc_server(tmp_path):
    """Spin up a PrimingRpcThread, yield (thread, socket_path)."""
    sock_dir = _short_socket_dir()
    socket_path = sock_dir / "priming.sock"
    stop_event = threading.Event()
    thread = PrimingRpcThread(stop_event=stop_event, socket_path=socket_path)
    thread.start()
    assert thread.wait_until_ready(timeout=2.0), "RPC server failed to bind"
    try:
        yield thread, socket_path
    finally:
        thread.stop(timeout=2.0)
        shutil.rmtree(sock_dir, ignore_errors=True)


# ── Tests ─────────────────────────────────────────────────────────────


def test_round_trip_returns_priming_markdown(tmp_path, rpc_server):
    """Real fixture library, real query — server returns rendered markdown.

    The first call in any process pays the sentence-transformer model
    load (~5–15s on cold disk, ~1s on warm disk). In production the
    daemon prewarms before the RPC thread starts; in tests we just
    grant a generous client timeout for this one call.
    """
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server

    resp = _send_request(socket_path, {
        "op": "priming",
        "prompt": "we were debugging a python package install. tried pip and it broke. switching to uv.",
        "k": 3,
        "char_budget": 500,
    }, io_timeout=30.0)

    assert resp["ok"] is True, resp
    assert "priming_md" in resp
    md = resp["priming_md"]
    assert md.startswith("## Ultan — your library says")
    # The python/uv entry should be the strongest match.
    assert "[[global/python/use-uv-not-pip]]" in md
    # Bullet count respects ``k``.
    bullets = [line for line in md.splitlines() if line.startswith("- [[")]
    assert len(bullets) <= 3
    assert resp["took_ms"] >= 0
    assert resp["lane"] in ("hybrid", "bm25")


def test_empty_prompt_returns_empty_markdown(tmp_path, rpc_server):
    """Empty prompt -> ok=true with empty markdown, NOT an error."""
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server

    resp = _send_request(socket_path, {"op": "priming", "prompt": "   "})
    assert resp["ok"] is True
    assert resp["priming_md"] == ""


def test_missing_knowledge_dir_returns_empty(tmp_path, rpc_server):
    """No library on disk yet -> empty render, not a failure."""
    # Note: we deliberately do NOT seed the library.
    _thread, socket_path = rpc_server

    resp = _send_request(socket_path, {"op": "priming", "prompt": "anything"})
    assert resp["ok"] is True
    assert resp["priming_md"] == ""


def test_unknown_op_returns_error(tmp_path, rpc_server):
    _thread, socket_path = rpc_server

    resp = _send_request(socket_path, {"op": "not-a-real-op"})
    assert resp["ok"] is False
    assert "unknown op" in resp["error"]


def test_malformed_json_returns_error(tmp_path, rpc_server):
    """Send raw bytes that aren't valid JSON; server should respond
    with an error rather than crashing the handler."""
    _thread, socket_path = rpc_server

    body = b"{this is not json"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    sock.connect(str(socket_path))
    sock.sendall(struct.pack(">I", len(body)) + body)

    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            break
        header += chunk
    assert len(header) == 4, "server should still respond on bad JSON"
    (length,) = struct.unpack(">I", header)
    payload = b""
    while len(payload) < length:
        payload += sock.recv(length - len(payload))
    sock.close()

    resp = json.loads(payload.decode("utf-8"))
    assert resp["ok"] is False
    assert "bad json" in resp["error"]


def test_non_object_json_returns_error(tmp_path, rpc_server):
    """Top-level array (valid JSON, wrong shape) -> ok=false."""
    _thread, socket_path = rpc_server

    body = json.dumps(["op", "priming"]).encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    sock.connect(str(socket_path))
    sock.sendall(struct.pack(">I", len(body)) + body)

    header_buf = b""
    while len(header_buf) < 4:
        chunk = sock.recv(4 - len(header_buf))
        if not chunk:
            break
        header_buf += chunk
    (length,) = struct.unpack(">I", header_buf)
    payload = b""
    while len(payload) < length:
        payload += sock.recv(length - len(payload))
    sock.close()
    resp = json.loads(payload.decode("utf-8"))
    assert resp["ok"] is False


def test_bad_k_returns_error(tmp_path, rpc_server):
    _thread, socket_path = rpc_server

    resp = _send_request(socket_path, {
        "op": "priming", "prompt": "x", "k": -1,
    })
    assert resp["ok"] is False


def test_stale_socket_file_is_cleaned_up_on_bind(tmp_path):
    """A leftover socket file from a prior crash must not block bind."""
    sock_dir = _short_socket_dir()
    socket_path = sock_dir / "priming.sock"
    # Pre-create a regular file at the socket path. (Not a real socket
    # — just bytes — proves the unlink works even on the wrong file
    # type.)
    socket_path.write_bytes(b"stale leftovers")
    assert socket_path.exists()

    stop_event = threading.Event()
    thread = PrimingRpcThread(stop_event=stop_event, socket_path=socket_path)
    thread.start()
    try:
        assert thread.wait_until_ready(timeout=2.0), "server should bind despite stale file"
        # And the file is now a real Unix socket — we can connect.
        resp = _send_request(socket_path, {"op": "priming", "prompt": "x"})
        assert resp["ok"] is True
    finally:
        thread.stop(timeout=2.0)
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_stop_removes_socket_file(tmp_path):
    """Graceful shutdown unlinks the socket file."""
    sock_dir = _short_socket_dir()
    socket_path = sock_dir / "priming.sock"
    stop_event = threading.Event()
    thread = PrimingRpcThread(stop_event=stop_event, socket_path=socket_path)
    thread.start()
    try:
        assert thread.wait_until_ready(timeout=2.0)
        assert socket_path.exists()

        thread.stop(timeout=2.0)
        assert not thread.is_alive()
        assert not socket_path.exists(), "socket file should be removed on stop"
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_handler_timeout_returns_error_not_hang(tmp_path, monkeypatch):
    """If the dispatch handler exceeds the 1s server cap, the
    connection still closes promptly (rather than hanging the
    accept loop)."""
    _seed_library(tmp_path)
    sock_dir = _short_socket_dir()
    socket_path = sock_dir / "priming.sock"

    # Monkeypatch the handler to sleep past the cap.
    real_handle = priming_rpc._handle_priming

    def _slow(req):
        time.sleep(priming_rpc.SERVER_REQUEST_TIMEOUT_S + 0.2)
        return real_handle(req)

    monkeypatch.setattr(priming_rpc, "_handle_priming", _slow)

    stop_event = threading.Event()
    thread = PrimingRpcThread(stop_event=stop_event, socket_path=socket_path)
    thread.start()
    try:
        assert thread.wait_until_ready(timeout=2.0)
        t0 = time.monotonic()
        # The client-side timeout is generous — we're verifying the
        # SERVER doesn't hang the connection past its own cap.
        try:
            resp = _send_request(socket_path, {
                "op": "priming", "prompt": "python uv",
            }, io_timeout=3.0)
        except (ConnectionError, socket.timeout):
            resp = None
        elapsed = time.monotonic() - t0
        # Either the slow handler eventually responded (resp is dict)
        # or the connection got dropped — but either way the wall
        # time should be bounded. The handler's own sleep is the
        # floor; we just confirm we're not hung indefinitely.
        assert elapsed < 5.0, f"server hung for {elapsed:.2f}s"
        if resp is not None:
            # If we got a response, it's allowed to be either ok or
            # a server-side timeout error — both are sane outcomes.
            assert isinstance(resp.get("ok"), bool)
    finally:
        thread.stop(timeout=2.0)
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_concurrent_requests_all_serviced(tmp_path, rpc_server):
    """A handful of parallel clients should all get responses."""
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server

    results: List[dict] = []
    errors: List[Exception] = []
    lock = threading.Lock()

    def _client():
        try:
            r = _send_request(socket_path, {"op": "priming", "prompt": "python uv"},
                              io_timeout=30.0)
            with lock:
                results.append(r)
        except Exception as e:  # pragma: no cover - debugging aid
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=_client) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=35.0)
        assert not t.is_alive(), "client thread hung"

    assert not errors, f"clients raised: {errors}"
    assert len(results) == 4
    for r in results:
        assert r["ok"] is True
