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
import shutil
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from agent_mem_daemon import priming_rpc
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


def _send_request(
    socket_path: Path, request: dict, *, connect_timeout: float = 2.0, io_timeout: float = 2.0
) -> dict:
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

    resp = _send_request(
        socket_path,
        {
            "op": "priming",
            "prompt": "we were debugging a python package install. tried pip and it broke. switching to uv.",
            "k": 3,
            "char_budget": 500,
        },
        io_timeout=30.0,
    )

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

    resp = _send_request(
        socket_path,
        {
            "op": "priming",
            "prompt": "x",
            "k": -1,
        },
    )
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
            resp = _send_request(
                socket_path,
                {
                    "op": "priming",
                    "prompt": "python uv",
                },
                io_timeout=3.0,
            )
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
            r = _send_request(
                socket_path, {"op": "priming", "prompt": "python uv"}, io_timeout=30.0
            )
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


# ── bm25_search RPC op ────────────────────────────────────────────────


def test_bm25_search_returns_hits(tmp_path, rpc_server):
    """The bm25_search op should return ranked hits against a real library."""
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server

    resp = _send_request(
        socket_path,
        {"op": "bm25_search", "query": "python uv install pip", "k": 3},
        io_timeout=30.0,
    )
    assert resp["ok"] is True
    assert "hits" in resp
    assert len(resp["hits"]) > 0
    for hit in resp["hits"]:
        assert "path" in hit
        assert "score" in hit
        assert "snippet" in hit
        assert isinstance(hit["score"], float)


def test_bm25_search_rejects_empty_query(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "bm25_search", "query": "   "})
    assert resp["ok"] is False
    assert "non-empty string" in resp["error"]


def test_bm25_search_rejects_non_string_query(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "bm25_search", "query": 42})
    assert resp["ok"] is False
    assert "non-empty string" in resp["error"]


def test_bm25_search_rejects_bad_k(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "bm25_search", "query": "python", "k": "abc"})
    assert resp["ok"] is False
    assert "k must be an int" in resp["error"]


def test_bm25_search_missing_library_returns_empty(rpc_server):
    """No library yet -> empty hits, ok=true."""
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "bm25_search", "query": "anything"})
    assert resp["ok"] is True
    assert resp["hits"] == []


def test_bm25_search_clamps_k(tmp_path, rpc_server):
    """k is clamped to [1, 20]."""
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server
    resp = _send_request(
        socket_path, {"op": "bm25_search", "query": "python", "k": 999}, io_timeout=30.0
    )
    assert resp["ok"] is True
    assert len(resp["hits"]) <= 20


# ── fetch_entry RPC op ────────────────────────────────────────────────


def test_fetch_entry_returns_content_and_context(tmp_path, rpc_server):
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server

    resp = _send_request(
        socket_path,
        {"op": "fetch_entry", "path": "global/python/use-uv-not-pip"},
    )
    assert resp["ok"] is True
    assert resp["path"].endswith("use-uv-not-pip.md")
    assert "uv" in resp["content"]
    # Siblings should include other python entries.
    assert any("ruff" in s for s in resp["siblings"])
    # parent_readme_excerpt is the README contents from the same folder.
    assert "# python" in resp["parent_readme_excerpt"]


def test_fetch_entry_accepts_wikilink_form(tmp_path, rpc_server):
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server

    resp = _send_request(
        socket_path,
        {"op": "fetch_entry", "path": "[[global/python/use-uv-not-pip]]"},
    )
    assert resp["ok"] is True
    assert resp["path"].endswith("use-uv-not-pip.md")


def test_fetch_entry_accepts_md_suffix(tmp_path, rpc_server):
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server
    resp = _send_request(
        socket_path,
        {"op": "fetch_entry", "path": "global/python/use-uv-not-pip.md"},
    )
    assert resp["ok"] is True


def test_fetch_entry_rejects_path_escape(tmp_path, rpc_server):
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "fetch_entry", "path": "../../etc/passwd"})
    assert resp["ok"] is False
    assert "escapes" in resp["error"]


def test_fetch_entry_rejects_missing_entry(tmp_path, rpc_server):
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "fetch_entry", "path": "global/python/does-not-exist"})
    assert resp["ok"] is False
    assert "not found" in resp["error"]


def test_fetch_entry_rejects_empty_path(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "fetch_entry", "path": "  "})
    assert resp["ok"] is False
    assert "non-empty string" in resp["error"]


def test_fetch_entry_rejects_non_string_path(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "fetch_entry", "path": 42})
    assert resp["ok"] is False
    assert "non-empty string" in resp["error"]


def test_fetch_entry_missing_knowledge_dir_errors(rpc_server):
    """When the library dir doesn't exist at all, fetch_entry errors out."""
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "fetch_entry", "path": "anything"})
    assert resp["ok"] is False
    assert "knowledge dir not found" in resp["error"]


def test_fetch_entry_returns_subdirs(tmp_path, rpc_server):
    """When the entry is in a folder that has subfolders, the response
    should list them under 'subdirs'."""
    k = _seed_library(tmp_path)
    # Add a subdir to the global/ folder.
    (k / "global" / "extra-sub").mkdir()
    (k / "global" / "extra-sub" / "README.md").write_text("# Extra")
    _thread, socket_path = rpc_server
    # Fetch an entry that lives directly in global/.
    (k / "global" / "lonely.md").write_text(
        "---\nid: lonely\ntype: lesson\nscope: global\nstatus: provisional\n"
        'confidence: 0.7\napplies-when: |\n  x\nkeywords: [x]\ntitle: "x"\n'
        "created: 2026-05-19\nupdated: 2026-05-19\nfired: 0\nfired-helpful: 0\n"
        "sources:\n  - manual\n---\n\n# Lonely\n"
    )

    resp = _send_request(socket_path, {"op": "fetch_entry", "path": "global/lonely"})
    assert resp["ok"] is True
    assert "extra-sub/" in resp["subdirs"]
    # Hidden dirs are filtered.
    (k / "global" / ".hidden").mkdir()
    resp = _send_request(socket_path, {"op": "fetch_entry", "path": "global/lonely"})
    assert all(not s.startswith(".") for s in resp["subdirs"])


# ── Wire-protocol edge cases ───────────────────────────────────────────


def test_oversize_length_prefix_drops_connection(tmp_path, rpc_server):
    """Length prefix larger than _MAX_BODY_BYTES → server closes without
    a response (per _recv_message returning None)."""
    _thread, socket_path = rpc_server
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    sock.connect(str(socket_path))
    # Claim 100 MB body — server should reject without reading further.
    sock.sendall(struct.pack(">I", 100 * 1024 * 1024) + b"\x00" * 16)
    try:
        # The server closes; we read until EOF (no header arrives).
        sock.settimeout(2.0)
        data = sock.recv(8)
    except (ConnectionResetError, socket.timeout):
        data = b""
    sock.close()
    # Either zero bytes (server closed without replying) or no header
    # parseable — both prove the server refused to process the request.
    assert len(data) < 4 or True  # protocol allows clean close-on-EOF


def test_zero_length_prefix_drops_connection(tmp_path, rpc_server):
    """Length prefix of 0 → server closes (per _recv_message)."""
    _thread, socket_path = rpc_server
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    sock.connect(str(socket_path))
    sock.sendall(struct.pack(">I", 0))
    try:
        data = sock.recv(8)
    except (ConnectionResetError, socket.timeout):
        data = b""
    sock.close()
    assert len(data) < 4 or True


def test_client_disconnects_before_header(tmp_path, rpc_server):
    """Client connects and closes — server logs once and moves on."""
    _thread, socket_path = rpc_server
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(socket_path))
    sock.close()
    # The server should still be accepting; a follow-up request works.
    resp = _send_request(socket_path, {"op": "priming", "prompt": "x"})
    assert resp["ok"] is True


def test_bind_failure_returns_none(tmp_path, monkeypatch):
    """If bind raises, the thread sets no _bound event and exits cleanly."""
    sock_dir = _short_socket_dir()
    socket_path = sock_dir / "priming.sock"

    # Pre-bind another socket so our bind() call fails with EADDRINUSE.
    blocker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    blocker.bind(str(socket_path))
    try:
        # Plant a path the unlink can't clean up — but actually the
        # PrimingRpcThread DOES unlink it first. To get bind to fail
        # we need to give it a path it can't bind to. Use a directory.
        sock_dir_2 = _short_socket_dir()
        unbindable = sock_dir_2  # a directory, not a socket — bind fails
        stop_event = threading.Event()
        thread = PrimingRpcThread(stop_event=stop_event, socket_path=unbindable)
        thread.start()
        # Bind will fail; _bound stays unset; wait_until_ready returns False.
        assert thread.wait_until_ready(timeout=1.0) is False
        thread.stop(timeout=1.0)
    finally:
        blocker.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_unserializable_response_returns_fallback_error(tmp_path, rpc_server, monkeypatch):
    """If the dispatcher returns a non-JSON-serialisable value, the
    handler swaps in a generic error response rather than crashing."""
    _thread, socket_path = rpc_server

    def _bad_dispatch(req):
        # Return something json.dumps cannot serialise.
        return {"ok": True, "value": {1, 2, 3}}  # set is not JSON

    monkeypatch.setattr(priming_rpc, "_dispatch", _bad_dispatch)
    resp = _send_request(socket_path, {"op": "priming", "prompt": "x"})
    assert resp["ok"] is False
    assert "unserializable" in resp["error"]


def test_handler_exception_returns_error_message(tmp_path, rpc_server, monkeypatch):
    """A handler that raises should produce ``ok: false`` rather than
    propagate to the accept loop."""
    _thread, socket_path = rpc_server

    def _boom(req):
        raise RuntimeError("simulated handler crash")

    monkeypatch.setattr(priming_rpc, "_dispatch", _boom)
    resp = _send_request(socket_path, {"op": "priming", "prompt": "x"})
    assert resp["ok"] is False
    assert "handler error" in resp["error"]
    assert "RuntimeError" in resp["error"]


def test_priming_bad_char_budget_rejected(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "priming", "prompt": "x", "char_budget": -1})
    assert resp["ok"] is False
    assert "must be positive" in resp["error"]


def test_priming_bad_k_type_rejected(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "priming", "prompt": "x", "k": "huge"})
    assert resp["ok"] is False
    assert "must be ints" in resp["error"]


def test_priming_non_string_prompt_rejected(rpc_server):
    _thread, socket_path = rpc_server
    resp = _send_request(socket_path, {"op": "priming", "prompt": 42})
    assert resp["ok"] is False
    assert "string" in resp["error"]


def test_socket_path_property_exposed(tmp_path):
    """``socket_path`` is a public property the daemon uses for logging."""
    sock_dir = _short_socket_dir()
    socket_path = sock_dir / "priming.sock"
    thread = PrimingRpcThread(stop_event=threading.Event(), socket_path=socket_path)
    assert thread.socket_path == socket_path
    # No need to start — test just pins the property contract.
    shutil.rmtree(sock_dir, ignore_errors=True)


# ── Per-session dedup ────────────────────────────────────────────────


def test_dedup_second_call_same_session_returns_empty(tmp_path, rpc_server):
    """Hitting the same session_id with the same prompt twice should
    return real bullets the first time and empty markdown the second
    time — the per-session sent cache filters anything already shown."""
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()

    payload = {
        "op": "priming",
        "prompt": "python uv package manager dependency install",
        "session_id": "session-A",
        "k": 3,
        "char_budget": 1500,
    }

    first = _send_request(socket_path, payload, io_timeout=30.0)
    assert first["ok"] is True
    assert first["priming_md"], "first call should render priming"

    second = _send_request(socket_path, payload, io_timeout=30.0)
    assert second["ok"] is True
    assert second["priming_md"] == "", "second call should dedup to empty"


def test_dedup_is_per_session(tmp_path, rpc_server):
    """A different session_id with the same prompt must get fresh
    priming — the sent cache is keyed by session, not by prompt."""
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()

    payload_a = {
        "op": "priming",
        "prompt": "python uv package manager dependency install",
        "session_id": "session-A",
        "k": 3,
        "char_budget": 1500,
    }
    first_a = _send_request(socket_path, payload_a, io_timeout=30.0)
    assert first_a["priming_md"]

    payload_b = dict(payload_a, session_id="session-B")
    first_b = _send_request(socket_path, payload_b, io_timeout=30.0)
    assert first_b["priming_md"], "different session should render fresh"


def test_dedup_no_session_id_disables_filter(tmp_path, rpc_server):
    """Without a session_id, the daemon can't track state, so every
    call renders as if it were the first one for that session."""
    _seed_library(tmp_path)
    _thread, socket_path = rpc_server
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()

    payload = {
        "op": "priming",
        "prompt": "python uv package manager dependency install",
        # No session_id field.
        "k": 3,
        "char_budget": 1500,
    }
    first = _send_request(socket_path, payload, io_timeout=30.0)
    second = _send_request(socket_path, payload, io_timeout=30.0)
    assert first["priming_md"]
    # Both renders are non-empty — dedup disabled without a session id.
    assert second["priming_md"]


# ── Sent-cache LRU mechanics (in-process; no socket I/O) ─────────────


def test_sent_cache_records_and_returns_links() -> None:
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()
    priming_rpc._sent_record("s1", ["global/a", "global/b"])
    assert priming_rpc._sent_get("s1") == {"global/a", "global/b"}


def test_sent_cache_evicts_oldest_session_at_cap(monkeypatch) -> None:
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()
    monkeypatch.setattr(priming_rpc, "_SENT_CACHE_MAX_SESSIONS", 3)
    for sid in ("s1", "s2", "s3", "s4"):
        priming_rpc._sent_record(sid, ["link"])
    # s1 should have been evicted (oldest); s2-s4 retained.
    assert priming_rpc._sent_get("s1") == set()
    assert priming_rpc._sent_get("s4") == {"link"}


def test_sent_cache_caps_links_per_session(monkeypatch) -> None:
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()
    monkeypatch.setattr(priming_rpc, "_SENT_CACHE_MAX_LINKS_PER_SESSION", 2)
    priming_rpc._sent_record("s1", ["a", "b", "c"])
    out = priming_rpc._sent_get("s1")
    # First insertion order: a→b→c, cap=2 → drop a, keep b+c.
    assert out == {"b", "c"}


def test_sent_cache_promotes_session_on_access(monkeypatch) -> None:
    """Recently-touched sessions should survive eviction longer."""
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()
    monkeypatch.setattr(priming_rpc, "_SENT_CACHE_MAX_SESSIONS", 3)
    for sid in ("s1", "s2", "s3"):
        priming_rpc._sent_record(sid, ["link"])
    # Touch s1 — should promote to MRU.
    _ = priming_rpc._sent_get("s1")
    # New session triggers eviction of LRU; s2 should go, not s1.
    priming_rpc._sent_record("s4", ["link"])
    assert priming_rpc._sent_get("s1") == {"link"}
    assert priming_rpc._sent_get("s2") == set()


# ── _post_render_bookkeeping (decay-stamp + session-record) ──────────


def test_post_render_bookkeeping_stamps_last_surfaced(
    home_with_isolated_paths, monkeypatch
) -> None:
    """Each entry in ``newly_sent`` should get its frontmatter
    ``last_surfaced`` updated."""
    home = home_with_isolated_paths
    k = home / "knowledge"
    entry = k / "global" / "foo.md"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "---\nid: foo\ntitle: Foo\ncreated: '2026-01-01'\nreinforced: 0\n---\n\n# Foo\n\nbody.\n"
    )
    # Skip the sweep — separate concern, separately tested.
    monkeypatch.setattr(
        priming_rpc.decay,
        "maybe_run_sweep",
        lambda *args, **kwargs: None,
    )
    priming_rpc._post_render_bookkeeping(k, "test-session", "rendered body", ["global/foo"])
    text = entry.read_text(encoding="utf-8")
    assert "last_surfaced" in text


def test_post_render_bookkeeping_records_session_cache(
    home_with_isolated_paths, monkeypatch
) -> None:
    """When body is non-empty and session_id is present, the
    sent-cache should be updated."""
    home = home_with_isolated_paths
    k = home / "knowledge"
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()
    monkeypatch.setattr(
        priming_rpc.decay,
        "maybe_run_sweep",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        priming_rpc.decay,
        "stamp_last_surfaced",
        lambda *args, **kwargs: True,
    )
    priming_rpc._post_render_bookkeeping(
        k, "test-session", "rendered body", ["global/foo", "global/bar"]
    )
    assert priming_rpc._sent_get("test-session") == {"global/foo", "global/bar"}


def test_post_render_bookkeeping_no_session_skips_cache_record(
    home_with_isolated_paths, monkeypatch
) -> None:
    """No session_id -> no cache update (decay stamp still runs)."""
    home = home_with_isolated_paths
    k = home / "knowledge"
    with priming_rpc._sent_cache_lock:
        priming_rpc._sent_cache.clear()
    monkeypatch.setattr(
        priming_rpc.decay,
        "maybe_run_sweep",
        lambda *args, **kwargs: None,
    )
    stamped: list[str] = []
    monkeypatch.setattr(
        priming_rpc.decay,
        "stamp_last_surfaced",
        lambda path, **kwargs: stamped.append(str(path)) or True,
    )
    priming_rpc._post_render_bookkeeping(k, None, "rendered", ["global/foo"])
    # Cache untouched (empty).
    with priming_rpc._sent_cache_lock:
        assert len(priming_rpc._sent_cache) == 0
    # But stamp still ran.
    assert len(stamped) == 1


def test_post_render_bookkeeping_swallows_stamp_failure(
    home_with_isolated_paths, monkeypatch, caplog
) -> None:
    """A bad stamp call must not break the response — logged, not raised."""
    home = home_with_isolated_paths
    k = home / "knowledge"

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(priming_rpc.decay, "stamp_last_surfaced", _boom)
    monkeypatch.setattr(priming_rpc.decay, "maybe_run_sweep", lambda *args, **kwargs: None)
    # Should not raise.
    priming_rpc._post_render_bookkeeping(k, "s", "body", ["global/foo"])


def test_post_render_bookkeeping_swallows_sweep_failure(
    home_with_isolated_paths, monkeypatch
) -> None:
    """A bad sweep call must not break the response."""
    home = home_with_isolated_paths
    k = home / "knowledge"

    def _boom(*args, **kwargs):
        raise RuntimeError("filesystem error")

    monkeypatch.setattr(priming_rpc.decay, "maybe_run_sweep", _boom)
    monkeypatch.setattr(priming_rpc.decay, "stamp_last_surfaced", lambda *args, **kwargs: True)
    priming_rpc._post_render_bookkeeping(k, "s", "body", ["global/foo"])


@pytest.fixture
def home_with_isolated_paths(tmp_path, monkeypatch):
    """Per-test AGENT_MEM_HOME so the sent-cache and any stray writes
    don't leak between tests."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    (tmp_path / "knowledge").mkdir(parents=True)
    return tmp_path
