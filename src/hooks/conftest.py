"""Shared pytest fixtures for hook tests.

Every hook follows the same shape: read JSON from stdin, write events
to ``${AGENT_MEM_HOME}/events.jsonl``, sometimes write a structured
JSON response to stdout. The fixtures here build the scaffolding once:

- ``short_home`` — a tmp ``AGENT_MEM_HOME`` under ``/tmp`` so the
  ``priming.sock`` path stays under macOS's 104-char AF_UNIX cap.
  pytest's own ``tmp_path`` (under ``/var/folders/.../pytest-...``) is
  already over 90 chars so anything beneath it can't host a socket.
  Tests that don't touch the priming socket can use the stdlib
  ``tmp_path`` fixture; tests that do (or might) should take this.

- ``isolated_home`` — sets ``AGENT_MEM_HOME`` to ``short_home`` via
  monkeypatch so anything calling ``Path.home()`` / reading the env
  var picks up the test dir. Also clears ``CLAUDE_INVOKED_BY`` so the
  recursion guard doesn't short-circuit.

- ``fake_rpc_server`` — context-manager factory wrapping the
  :class:`_FakeRpcServer` so tests don't have to remember the
  start/stop/join dance, and the well-known
  ``threading.Thread._stop`` collision (Thread defines a ``_stop``
  method) can't bite again.

- ``seed_lib`` — populates ``<home>/knowledge`` with a tiny library
  the lexical-fallback search can hit. Shared between priming and
  end-to-end tests.

- ``hook_runner`` — subprocess helper that invokes any hyphenated
  hook script with crafted stdin and returns the
  ``CompletedProcess``. Encapsulates the env-passing pattern (clear
  CLAUDE_INVOKED_BY, inherit just enough for Python to find its
  stdlib, allow per-test overrides).
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Callable, ContextManager, Iterator

import pytest

_THIS_DIR = Path(__file__).resolve().parent


# ── Short-path home dir ──────────────────────────────────────────────


@pytest.fixture
def short_home() -> Iterator[Path]:
    """``AGENT_MEM_HOME`` candidate dir under ``/tmp`` (short path).

    macOS caps AF_UNIX socket paths at ~104 chars; pytest's tmp_path
    is too long for the socket to live anywhere underneath it. Tests
    that need to bind a fake priming socket take this fixture.
    """
    d = Path(tempfile.mkdtemp(prefix="ult-home-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def isolated_home(short_home: Path, monkeypatch) -> Path:
    """Pin AGENT_MEM_HOME for the test and clear the recursion guard.

    Returns the same path ``short_home`` resolves to so tests can
    inspect post-conditions (events.jsonl, pending-nudges.md, etc.).
    """
    monkeypatch.setenv("AGENT_MEM_HOME", str(short_home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    return short_home


# ── Library seeding ──────────────────────────────────────────────────


def _entry(
    path: Path,
    *,
    title: str,
    applies_when: str,
    keywords: list,
    reinforced: int = 0,
    body: str = "",
) -> None:
    """Write a minimal lesson entry. Mirrors the shape the lexical
    fallback expects (frontmatter with keywords + applies-when, body
    with at least a heading).
    """
    lines = [
        "---",
        f"id: {path.stem}",
        "type: lesson",
        "scope: global",
        "status: provisional",
        "confidence: 0.7",
        "applies-when: |",
        f"  {applies_when}",
        "keywords: [" + ", ".join(keywords) + "]",
        f'title: "{title}"',
    ]
    if reinforced:
        lines.append(f"reinforced: {reinforced}")
    lines += ["---", "", body or f"# {title}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def seed_library(home: Path) -> Path:
    """Seed a tiny knowledge dir under ``home`` and return its path.

    Shared by the priming, end-to-end, and priming-socket hook tests so
    they all exercise one library shape.
    """
    k = home / "knowledge"
    for sub in ("global/python", "global/git"):
        (k / sub).mkdir(parents=True, exist_ok=True)
        (k / sub / "README.md").write_text(f"# {sub}\n", encoding="utf-8")
    (k / "README.md").write_text("# knowledge\n", encoding="utf-8")
    # log.md as the blocker-cache sentinel
    (k / "log.md").write_text("# Build Log\n", encoding="utf-8")

    _entry(
        k / "global" / "python" / "use-uv-not-pip.md",
        title="Always use uv for python",
        applies_when="installing python deps",
        keywords=["python", "uv", "pip"],
        reinforced=3,
    )
    _entry(
        k / "global" / "python" / "type-hints.md",
        title="Use type hints",
        applies_when="writing python functions",
        keywords=["python", "types"],
    )
    _entry(
        k / "global" / "git" / "no-force-push.md",
        title="Never force-push",
        applies_when="pushing to git remotes",
        keywords=["git", "push"],
    )
    return k


@pytest.fixture
def seed_lib() -> Callable[[Path], Path]:
    """Fixture wrapper around :func:`seed_library`."""
    return seed_library


# ── Flush-spawning hook helpers (session_end / pre_compact) ──────────


class FakePopen:
    """Stand-in for ``subprocess.Popen`` so the flush-spawning hooks never
    actually launch flush.py during tests."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.returncode = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fresh_hook(monkeypatch, home: Path, module_name: str) -> ModuleType:
    """Import a flush-spawning hook module with ``AGENT_MEM_HOME`` pinned and
    ``_flush_spawn.subprocess.Popen`` stubbed so no real flush is spawned.

    No ``sys.modules`` surgery: ``config.get_config()`` reads
    ``AGENT_MEM_HOME`` at call time, so a plain import already picks up the
    test env.
    """
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    import _flush_spawn

    class _ShimSubprocess:
        Popen = staticmethod(FakePopen)
        DEVNULL = _flush_spawn.subprocess.DEVNULL
        CREATE_NO_WINDOW = getattr(_flush_spawn.subprocess, "CREATE_NO_WINDOW", 0)

    monkeypatch.setattr(_flush_spawn, "subprocess", _ShimSubprocess)
    return importlib.import_module(module_name)


def drive_stdin(monkeypatch, module: ModuleType, payload: dict) -> None:
    """Feed ``payload`` as JSON on stdin and call the hook module's main()."""
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    module.main()


# ── Fake priming-socket RPC server ───────────────────────────────────


class FakeRpcServer(threading.Thread):
    """Minimal length-prefixed-JSON server used to mock the daemon.

    Identical wire shape to ``daemon/agent_mem_daemon/priming_rpc.py``:
    a 4-byte big-endian length prefix followed by a UTF-8 JSON body.

    The handler is a callable ``(request_dict) -> response_dict`` so
    individual tests can canned-respond, raise, or sleep.

    NOTE: never name an instance attribute ``_stop`` on a Thread
    subclass — ``threading.Thread._stop`` is an internal method that
    ``Thread.join()`` calls during cleanup. Shadowing it with an Event
    makes join() raise ``TypeError: 'Event' object is not callable``.
    """

    def __init__(self, socket_path: Path, handler: Callable[[dict], dict]):
        super().__init__(daemon=True, name="fake-priming-rpc")
        self._socket_path = socket_path
        self._handler = handler
        self._server: socket.socket | None = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()

    def start_and_wait(self, timeout: float = 2.0) -> None:
        self.start()
        assert self._ready.wait(timeout=timeout), "fake server failed to bind"

    def _bind(self) -> socket.socket:
        if self._socket_path.exists() or self._socket_path.is_symlink():
            try:
                self._socket_path.unlink()
            except OSError:
                pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(self._socket_path))
        s.listen(16)
        s.settimeout(0.2)
        return s

    def run(self) -> None:
        self._server = self._bind()
        self._ready.set()
        try:
            while not self._stop_event.is_set():
                try:
                    conn, _ = self._server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    conn.settimeout(5.0)
                    header = b""
                    while len(header) < 4:
                        chunk = conn.recv(4 - len(header))
                        if not chunk:
                            break
                        header += chunk
                    if len(header) < 4:
                        conn.close()
                        continue
                    (length,) = struct.unpack(">I", header)
                    payload = b""
                    while len(payload) < length:
                        chunk = conn.recv(length - len(payload))
                        if not chunk:
                            break
                        payload += chunk
                    try:
                        req = json.loads(payload.decode("utf-8"))
                    except Exception:
                        req = {}
                    try:
                        resp = self._handler(req)
                    except Exception as e:
                        resp = {"ok": False, "error": f"handler raised: {e}"}
                    body = json.dumps(resp).encode("utf-8")
                    try:
                        conn.sendall(struct.pack(">I", len(body)) + body)
                    except OSError:
                        pass
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        finally:
            try:
                if self._server:
                    self._server.close()
            except OSError:
                pass
            try:
                if self._socket_path.exists() or self._socket_path.is_symlink():
                    self._socket_path.unlink()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
        self.join(timeout=2.0)


@pytest.fixture
def fake_rpc_server() -> Callable[[Path, Callable[[dict], dict]], ContextManager]:
    """Factory: ``with fake_rpc_server(path, handler) as srv: ...``.

    Hides the start/stop dance behind a context manager so every test
    that mocks the daemon gets the same cleanup behaviour.
    """

    @contextmanager
    def _factory(socket_path: Path, handler: Callable[[dict], dict]):
        srv = FakeRpcServer(socket_path, handler)
        srv.start_and_wait()
        try:
            yield srv
        finally:
            srv.stop()

    return _factory


# ── Subprocess hook runner ───────────────────────────────────────────


@pytest.fixture
def hook_runner() -> Callable[..., subprocess.CompletedProcess]:
    """Subprocess-driven hook invocation.

    Usage:
        result = hook_runner("user-prompt-submit.py", payload, env={...})

    Inherits the minimum env so Python can find its stdlib, then
    overlays whatever the test wants. Always clears CLAUDE_INVOKED_BY
    (unless the test deliberately sets it) so the recursion guard
    doesn't silently short-circuit.
    """

    def _run(
        hook_filename: str,
        stdin_payload: dict,
        *,
        env: dict | None = None,
        timeout: float = 10.0,
    ) -> subprocess.CompletedProcess:
        script = _THIS_DIR / hook_filename
        out_env = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
        for k in ("HOME", "PYTHONPATH", "VIRTUAL_ENV", "PATH", "PYTHONHOME"):
            if k in os.environ:
                out_env[k] = os.environ[k]
        if env:
            out_env.update(env)
        if env is None or "CLAUDE_INVOKED_BY" not in env:
            out_env.pop("CLAUDE_INVOKED_BY", None)
        return subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(stdin_payload),
            capture_output=True,
            text=True,
            env=out_env,
            timeout=timeout,
        )

    return _run
