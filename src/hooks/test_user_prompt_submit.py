"""Tests for the pending-nudges read+budget logic that
``user-prompt-submit.py`` depends on, plus an end-to-end test of the
hook itself (driven via subprocess so the stdin/stdout protocol path
is exercised verbatim).

Run from the ``src/`` directory with::

    uv run python -m pytest hooks/ -q

The hook script uses a hyphenated filename (``user-prompt-submit.py``)
because that matches what Claude Code's settings.json conventions
expect; we exercise it via subprocess rather than importing.
"""

from __future__ import annotations

import json
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# Make ``_nudges`` importable. pytest runs from src/, so hooks/ is the
# obvious place to add to sys.path. We do it once at module load.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import _nudges  # noqa: E402
import _priming_client  # noqa: E402

HOOK_SCRIPT = _THIS_DIR / "user-prompt-submit.py"


# ── Helpers for the priming/socket tests ─────────────────────────────


def _short_sock_dir() -> Path:
    """macOS caps AF_UNIX paths at ~104 chars; pytest's tmp_path is
    too long for AGENT_MEM_HOME to be both the socket parent AND a
    nested test dir. Use /tmp for the socket parent."""
    return Path(tempfile.mkdtemp(prefix="ult-hook-"))


def _seed_lib(root: Path) -> Path:
    """Tiny library so the local-lexical fallback (and a real daemon
    if one were running) has something to match. Mirrors the minimum
    shape used in tests/test_priming.py over in the daemon repo."""
    k = root / "knowledge"
    for sub in ("global/python", "global/git"):
        (k / sub).mkdir(parents=True, exist_ok=True)
        (k / sub / "README.md").write_text(f"# {sub}\n", encoding="utf-8")
    (k / "README.md").write_text("# knowledge\n", encoding="utf-8")

    def _entry(
        path: Path,
        *,
        title: str,
        applies_when: str,
        keywords: list,
        reinforced: int = 0,
        body: str = "",
    ) -> None:
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


class _FakeRpcServer(threading.Thread):
    """Minimal length-prefixed-JSON server used to mock the daemon.

    The handler is a callable ``(request_dict) -> response_dict`` so
    individual tests can canned-respond, raise, or sleep."""

    def __init__(self, socket_path: Path, handler):
        super().__init__(daemon=True, name="fake-priming-rpc")
        self._socket_path = socket_path
        self._handler = handler
        self._server: socket.socket | None = None
        self._stop = threading.Event()
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
            while not self._stop.is_set():
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
        self._stop.set()
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
        self.join(timeout=2.0)


# ── pure-Python tests of _nudges (fast, no subprocess) ───────────────


def test_parse_nudges_empty():
    assert _nudges.parse_nudges("") == []
    assert _nudges.parse_nudges("\n\n   \n") == []


def test_parse_nudges_three_blocks():
    body = (
        "---\nid: a1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/a\n---\nfirst\n"
        "---\nid: b2\ncreated: 2026-05-19T00:00:01+00:00\nlesson: l/b\n---\nsecond\n"
        "---\nid: c3\ncreated: 2026-05-19T00:00:02+00:00\nlesson: l/c\n---\nthird\n"
    )
    out = _nudges.parse_nudges(body)
    assert len(out) == 3
    assert out[0].id == "a1" and out[0].text == "first"
    assert out[2].lesson == "l/c"


def test_take_nudges_empty_file_returns_nothing(tmp_path: Path):
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    selected, consumed = _nudges.take_nudges(
        "s1",
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert selected == []
    assert consumed == 0
    # No state file should have been written for a no-op.
    assert not state_path.exists()


def test_take_nudges_missing_file_returns_nothing(tmp_path: Path):
    nudges_path = tmp_path / "does-not-exist.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    selected, consumed = _nudges.take_nudges(
        "s1",
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert selected == []
    assert consumed == 0


def test_take_nudges_budget_three_per_session(tmp_path: Path):
    """Three nudges queued, budget 1/turn, 3/session — across four turns
    we should see 1+1+1+0."""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"

    # Write three nudges all at once. The hook clears on first read, so
    # we rewrite the file before each successive turn to simulate the
    # daemon producing more — except for the third+ turn where we test
    # the per-session cap kicks in.
    three_blocks = (
        "---\nid: a1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/a\n---\nA text\n"
        "---\nid: b2\ncreated: 2026-05-19T00:00:01+00:00\nlesson: l/b\n---\nB text\n"
        "---\nid: c3\ncreated: 2026-05-19T00:00:02+00:00\nlesson: l/c\n---\nC text\n"
    )
    nudges_path.write_text(three_blocks, encoding="utf-8")
    # First turn: only 1 emitted (per-turn budget), file is consumed.
    sel1, consumed1 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert len(sel1) == 1
    assert sel1[0].id == "a1"
    assert consumed1 == 1
    # The file is cleared (renamed to .consumed).
    assert not nudges_path.exists()
    assert (tmp_path / "pending-nudges.md.consumed").exists()

    # Daemon writes more nudges before turn 2.
    nudges_path.write_text(
        "---\nid: d4\ncreated: 2026-05-19T00:00:03+00:00\nlesson: l/d\n---\nD text\n"
        "---\nid: e5\ncreated: 2026-05-19T00:00:04+00:00\nlesson: l/e\n---\nE text\n",
        encoding="utf-8",
    )
    sel2, consumed2 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert len(sel2) == 1
    assert sel2[0].id == "d4"
    assert consumed2 == 2

    # Turn 3: one more queued, budget allows.
    nudges_path.write_text(
        "---\nid: f6\ncreated: 2026-05-19T00:00:05+00:00\nlesson: l/f\n---\nF text\n",
        encoding="utf-8",
    )
    sel3, consumed3 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert len(sel3) == 1
    assert sel3[0].id == "f6"
    assert consumed3 == 3

    # Turn 4: more queued, but the per-session cap is exhausted.
    nudges_path.write_text(
        "---\nid: g7\ncreated: 2026-05-19T00:00:06+00:00\nlesson: l/g\n---\nG text\n",
        encoding="utf-8",
    )
    sel4, consumed4 = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert sel4 == []
    assert consumed4 == 3  # unchanged
    # The file is still cleared — we don't let a backlog accumulate.
    assert not nudges_path.exists()


def test_take_nudges_clears_file_even_when_over_budget(tmp_path: Path):
    """A session past its budget should still drain the file so unrelated
    sessions don't see stale content. (Today's design: one nudge file
    shared by all sessions — first-come first-serve.)"""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    # Prime state to "already at budget".
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"consumed": 3, "updated": 0}), encoding="utf-8")
    nudges_path.write_text(
        "---\nid: x1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/x\n---\nover-budget text\n",
        encoding="utf-8",
    )
    selected, consumed = _nudges.take_nudges(
        "s1", _nudges_path=nudges_path, _budget_state_path=state_path
    )
    assert selected == []
    assert consumed == 3
    assert not nudges_path.exists()  # cleared


def test_take_nudges_filters_cross_project_and_requeues(tmp_path: Path):
    """A vol-predictor nudge surfaced in an agent-mem session must be
    skipped AND re-queued for a future session in the matching project.
    Global and current-project nudges are delivered normally."""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    blocks = [
        # other project — should be filtered out and re-queued
        "---\nid: a1\ncreated: 2026-05-21T00:00:00+00:00\n"
        "lesson: projects/vol-predictor/foo.md\n---\nVOL text\n",
        # global — should always pass
        "---\nid: a2\ncreated: 2026-05-21T00:00:00+00:00\n"
        "lesson: global/bar.md\n---\nGLOBAL text\n",
        # current project — should pass
        "---\nid: a3\ncreated: 2026-05-21T00:00:00+00:00\n"
        "lesson: projects/agent-mem/baz.md\n---\nAGENTMEM text\n",
    ]
    nudges_path.write_text("\n".join(blocks), encoding="utf-8")

    selected, _consumed = _nudges.take_nudges(
        "s1",
        per_turn_budget=5,
        current_project_slug="agent-mem",
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    ids = [n.id for n in selected]
    assert "a1" not in ids, "cross-project nudge should NOT be delivered"
    assert ids == ["a2", "a3"], f"order preserved among eligible nudges: {ids}"

    # The vol-predictor nudge should have been written back to the file
    # so a future session in that project can claim it.
    assert nudges_path.exists(), "re-queue should re-create the nudges file"
    requeued_body = nudges_path.read_text(encoding="utf-8")
    assert "a1" in requeued_body
    assert "a2" not in requeued_body
    assert "a3" not in requeued_body


def test_take_nudges_uses_alias_map_to_match_slug_to_bucket(tmp_path: Path, monkeypatch):
    """Bucket ``agent-mem`` declares its canonical slug as
    ``github.com-nickroci-ultan``; a session with that slug should
    then deliver the agent-mem nudge. File shape is
    ``{<bucket>: <canonical-slug>}``."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    aliases = tmp_path / "project-aliases.json"
    aliases.write_text('{"agent-mem": "github.com-nickroci-ultan"}', encoding="utf-8")

    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    blocks = [
        # agent-mem entry — should pass once the alias is honoured
        "---\nid: a1\ncreated: 2026-05-21T00:00:00+00:00\n"
        "lesson: projects/agent-mem/baz.md\n---\nAGENTMEM text\n",
        # vol-predictor entry — should still be filtered out
        "---\nid: a2\ncreated: 2026-05-21T00:00:00+00:00\n"
        "lesson: projects/vol-predictor/foo.md\n---\nVOL text\n",
    ]
    nudges_path.write_text("\n".join(blocks), encoding="utf-8")

    selected, _ = _nudges.take_nudges(
        "s1",
        per_turn_budget=5,
        current_project_slug="github.com-nickroci-ultan",
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert [n.id for n in selected] == ["a1"], "alias should let slug match bucket"
    requeued = nudges_path.read_text(encoding="utf-8")
    assert "a2" in requeued and "a1" not in requeued


def test_take_nudges_no_slug_is_permissive(tmp_path: Path):
    """When the session has no project context (no slug derivable from
    cwd) we deliver every queued nudge — better than letting it sit
    forever waiting on a session that never comes."""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    nudges_path.write_text(
        "---\nid: a1\ncreated: 2026-05-21T00:00:00+00:00\n"
        "lesson: projects/vol-predictor/foo.md\n---\nVOL text\n",
        encoding="utf-8",
    )
    selected, _ = _nudges.take_nudges(
        "s1",
        per_turn_budget=5,
        current_project_slug=None,
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert [n.id for n in selected] == ["a1"]


def test_render_context_includes_count_and_paths():
    nudges = [
        _nudges.Nudge(id="a", created="t", lesson="l/a", text="alpha text"),
        _nudges.Nudge(id="b", created="t", lesson="l/b", text="beta text"),
    ]
    rendered = _nudges.render_context(nudges)
    assert "2 relevant" in rendered
    assert "alpha text" in rendered
    assert "beta text" in rendered
    assert "[[l/a]]" in rendered
    assert "[[l/b]]" in rendered
    assert "user has not been asked" in rendered


def test_render_context_empty_returns_empty():
    assert _nudges.render_context([]) == ""


def test_per_turn_budget_overridable(tmp_path: Path):
    """If a caller passes per_turn_budget=3 they get up to 3 at once
    (still capped by per-session budget)."""
    nudges_path = tmp_path / "pending-nudges.md"
    state_path = tmp_path / "state" / "nudge-budget-s1.json"
    nudges_path.write_text(
        "---\nid: a1\ncreated: t\nlesson: l/a\n---\nA\n"
        "---\nid: b2\ncreated: t\nlesson: l/b\n---\nB\n"
        "---\nid: c3\ncreated: t\nlesson: l/c\n---\nC\n",
        encoding="utf-8",
    )
    sel, consumed = _nudges.take_nudges(
        "s1",
        per_turn_budget=10,
        per_session_budget=3,
        _nudges_path=nudges_path,
        _budget_state_path=state_path,
    )
    assert len(sel) == 3
    assert consumed == 3


# ── End-to-end via subprocess (hook script) ───────────────────────────


def _run_hook(stdin_payload: dict, env_overrides: dict) -> subprocess.CompletedProcess:
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        # Belt-and-braces: ensure recursion guard is OFF for the test.
        # If a previous test happened to set it in the parent env we'd
        # short-circuit the hook entirely.
    }
    # Inherit just enough so Python can find its stdlib.
    import os

    for k in ("HOME", "PYTHONPATH", "VIRTUAL_ENV", "PATH", "PYTHONHOME"):
        if k in os.environ:
            env[k] = os.environ[k]
    env.update(env_overrides)
    env.pop("CLAUDE_INVOKED_BY", None)
    return subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def test_hook_emits_nothing_when_no_nudges(tmp_path: Path):
    res = _run_hook(
        {"session_id": "s1", "cwd": "/tmp", "prompt": "hello"},
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    # No additionalContext should be printed.
    assert res.stdout.strip() == "" or "additionalContext" not in res.stdout


def test_hook_emits_one_nudge_then_consumes(tmp_path: Path):
    nudges_path = tmp_path / "pending-nudges.md"
    nudges_path.write_text(
        "---\nid: a1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/a\n---\nfirst nudge text\n"
        "---\nid: b2\ncreated: 2026-05-19T00:00:01+00:00\nlesson: l/b\n---\nsecond nudge text\n",
        encoding="utf-8",
    )

    res = _run_hook(
        {"session_id": "s1", "cwd": "/tmp", "prompt": "do thing"},
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    stdout = res.stdout.strip()
    assert stdout, "hook should emit additionalContext when nudges queued"
    output = json.loads(stdout)
    ctx = output["hookSpecificOutput"]["additionalContext"]
    # Only ONE nudge per turn.
    assert "first nudge text" in ctx
    assert "second nudge text" not in ctx
    # Nudges file should be cleared (renamed to .consumed).
    assert not nudges_path.exists()
    assert (tmp_path / "pending-nudges.md.consumed").exists()
    # Budget state file written.
    budget = tmp_path / "state" / "nudge-budget-s1.json"
    assert budget.exists()
    data = json.loads(budget.read_text(encoding="utf-8"))
    assert data["consumed"] == 1


def test_hook_skips_nudges_when_no_session_id(tmp_path: Path):
    nudges_path = tmp_path / "pending-nudges.md"
    nudges_path.write_text(
        "---\nid: a1\ncreated: t\nlesson: l/a\n---\norphan\n",
        encoding="utf-8",
    )
    res = _run_hook(
        {"cwd": "/tmp", "prompt": "hi"},  # no session_id
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0
    assert "additionalContext" not in res.stdout
    # Without session_id we don't drain — leave the file intact.
    assert nudges_path.exists()


# ── Priming client (unit, in-process) ─────────────────────────────────


def test_get_priming_empty_prompt_returns_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    assert _priming_client.get_priming("") == ""
    assert _priming_client.get_priming("   ") == ""


def test_get_priming_no_socket_no_library_returns_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    assert _priming_client.get_priming("python uv install") == ""


def test_get_priming_local_fallback_when_socket_missing(tmp_path: Path, monkeypatch):
    """Daemon down (no socket file), but knowledge dir exists —
    fallback should still return rendered markdown."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    _seed_lib(tmp_path)

    out = _priming_client.get_priming("python uv pip install package")
    assert out.startswith("## Ultan — your library says"), out
    assert "[[global/python/use-uv-not-pip]]" in out
    # Reinforced count should propagate.
    assert "(×3)" in out


def test_get_priming_socket_success_short_circuits_fallback(tmp_path: Path, monkeypatch):
    """When the socket responds ok=true, we use that payload and don't
    fall back even if it differs from what local search would produce."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    _seed_lib(tmp_path)

    sock_dir = _short_sock_dir()
    socket_path = sock_dir / "priming.sock"
    monkeypatch.setattr(_priming_client, "_priming_socket_path", lambda: socket_path)

    canned = "## Library priming (you may know more about this than you think)\n\n- [[FROM_DAEMON]] — canned\n\n*Tip-of-the-tongue? `/ultan-advisor <q>` pulls the full entry.*\n"
    server = _FakeRpcServer(
        socket_path, lambda req: {"ok": True, "priming_md": canned, "took_ms": 5, "lane": "hybrid"}
    )
    try:
        server.start_and_wait()
        out = _priming_client.get_priming("python uv")
        assert out == canned, "should pass through daemon payload verbatim"
    finally:
        server.stop()
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_get_priming_socket_timeout_falls_back_to_local(tmp_path: Path, monkeypatch):
    """Slow socket past the budget -> client gives up, uses local
    search. The total wall time stays under the budget + the local
    fallback's runtime."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    _seed_lib(tmp_path)

    sock_dir = _short_sock_dir()
    socket_path = sock_dir / "priming.sock"
    monkeypatch.setattr(_priming_client, "_priming_socket_path", lambda: socket_path)

    def _slow(req):
        time.sleep(2.0)  # well past the 200ms budget
        return {"ok": True, "priming_md": "should never be seen", "took_ms": 2000, "lane": "hybrid"}

    server = _FakeRpcServer(socket_path, _slow)
    try:
        server.start_and_wait()
        t0 = time.monotonic()
        # Tight budget — make sure we abandon the socket and fall back.
        out = _priming_client.get_priming("python uv pip", total_budget_ms=100)
        elapsed = time.monotonic() - t0

        # The fallback should fire and produce real bullets keyed on the
        # local lexical search.
        assert "## Ultan — your library says" in out
        assert "[[global/python/use-uv-not-pip]]" in out
        # Sanity: we did not wait for the slow server.
        assert elapsed < 1.0, f"client waited {elapsed:.2f}s; should've bailed at 100ms"
    finally:
        server.stop()
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_get_priming_socket_ok_false_falls_back(tmp_path: Path, monkeypatch):
    """ok=false from the server -> use local fallback."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    _seed_lib(tmp_path)

    sock_dir = _short_sock_dir()
    socket_path = sock_dir / "priming.sock"
    monkeypatch.setattr(_priming_client, "_priming_socket_path", lambda: socket_path)

    server = _FakeRpcServer(socket_path, lambda req: {"ok": False, "error": "nope"})
    try:
        server.start_and_wait()
        out = _priming_client.get_priming("python uv")
        # Local fallback fired.
        assert "## Ultan — your library says" in out
        assert "[[global/python/use-uv-not-pip]]" in out
    finally:
        server.stop()
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_get_priming_local_fallback_is_fast(tmp_path: Path, monkeypatch):
    """The pure-stdlib fallback must stay under 100 ms on a tiny library."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    _seed_lib(tmp_path)

    # First call may pay a tiny disk-warm cost; measure the second.
    _priming_client.get_priming("warm-up")
    t0 = time.monotonic()
    out = _priming_client.get_priming("python uv pip install")
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert "## Ultan — your library says" in out
    assert elapsed_ms < 100.0, f"local fallback took {elapsed_ms:.1f}ms (budget 100ms)"


def test_get_priming_socket_round_trip_is_fast(tmp_path: Path, monkeypatch):
    """Happy path with a fast fake server: total wall time well under
    the daemon-served 200 ms budget."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    _seed_lib(tmp_path)

    sock_dir = _short_sock_dir()
    socket_path = sock_dir / "priming.sock"
    monkeypatch.setattr(_priming_client, "_priming_socket_path", lambda: socket_path)

    server = _FakeRpcServer(
        socket_path,
        lambda req: {
            "ok": True,
            "priming_md": "## Ultan — your library says\n\n- [[x]]\n\n*Tip*\n",
            "took_ms": 1,
            "lane": "hybrid",
        },
    )
    try:
        server.start_and_wait()
        # Warm-up to dodge interpreter / import jitter on first read.
        _priming_client.get_priming("warm-up")
        t0 = time.monotonic()
        out = _priming_client.get_priming("python uv")
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        assert out.startswith("## Ultan — your library says")
        assert elapsed_ms < 200.0, f"socket round trip took {elapsed_ms:.1f}ms (budget 200ms)"
    finally:
        server.stop()
        shutil.rmtree(sock_dir, ignore_errors=True)


# ── Hook end-to-end with the socket path ──────────────────────────────


@pytest.fixture
def short_home():
    """AGENT_MEM_HOME under /tmp so the priming.sock path stays under
    the macOS AF_UNIX 104-char cap. pytest's own tmp_path is too long."""
    d = Path(tempfile.mkdtemp(prefix="ult-home-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_hook_emits_priming_from_socket(short_home: Path):
    """Subprocess hook -> talks to a fake server -> emits daemon's
    canned priming as additionalContext."""
    _seed_lib(short_home)

    canned_md = (
        "## Ultan — your library says (cite or follow when applicable)\n\n"
        "- [[global/python/use-uv-not-pip]] (×3) — installing python deps\n\n"
        "*Wikilinks resolve to real entries. Use the `ultan-search` skill to read one (returns content + sibling entries + subfolders + parent README so you can traverse), or `/ultan-advisor <question>` to have Sonnet + Opus intelligently synthesise across multiple entries.*\n"
    )

    # The hook's _priming_client resolves the socket via AGENT_MEM_HOME.
    home_sock = short_home / "priming.sock"
    server = _FakeRpcServer(
        home_sock,
        lambda req: {"ok": True, "priming_md": canned_md, "took_ms": 4, "lane": "hybrid"},
    )
    try:
        server.start_and_wait()
        res = _run_hook(
            {"session_id": "s1", "cwd": "/tmp", "prompt": "python uv install"},
            env_overrides={"AGENT_MEM_HOME": str(short_home)},
        )
        assert res.returncode == 0, res.stderr
        stdout = res.stdout.strip()
        assert stdout, f"expected priming injection, got nothing. stderr={res.stderr}"
        output = json.loads(stdout)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert "## Ultan — your library says" in ctx
        assert "use-uv-not-pip" in ctx
    finally:
        server.stop()


def test_hook_combines_priming_and_nudges(short_home: Path):
    """Both halves present — priming from the socket, then nudges,
    visually separated by ``## Active nudges``."""
    _seed_lib(short_home)

    nudges_path = short_home / "pending-nudges.md"
    nudges_path.write_text(
        "---\nid: n1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/n\n---\nnudge body text\n",
        encoding="utf-8",
    )

    home_sock = short_home / "priming.sock"
    canned_md = (
        "## Ultan — your library says (cite or follow when applicable)\n\n"
        "- [[global/python/use-uv-not-pip]] — installing python deps\n\n"
        "*Wikilinks resolve to real entries. Use the `ultan-search` skill to read one (returns content + sibling entries + subfolders + parent README so you can traverse), or `/ultan-advisor <question>` to have Sonnet + Opus intelligently synthesise across multiple entries.*\n"
    )
    server = _FakeRpcServer(
        home_sock,
        lambda req: {"ok": True, "priming_md": canned_md, "took_ms": 3, "lane": "hybrid"},
    )
    try:
        server.start_and_wait()
        res = _run_hook(
            {"session_id": "s1", "cwd": "/tmp", "prompt": "python uv install"},
            env_overrides={"AGENT_MEM_HOME": str(short_home)},
        )
        assert res.returncode == 0, res.stderr
        output = json.loads(res.stdout.strip())
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert "## Ultan — your library says" in ctx
        assert "## Active nudges" in ctx
        assert "nudge body text" in ctx
        # Priming MUST come before nudges in the output.
        assert ctx.index("## Ultan — your library says") < ctx.index("## Active nudges")
    finally:
        server.stop()


def test_hook_falls_back_when_daemon_down(short_home: Path):
    """No socket at all, but knowledge dir has entries — the hook's
    in-process fallback should still emit priming."""
    _seed_lib(short_home)
    res = _run_hook(
        {"session_id": "s1", "cwd": "/tmp", "prompt": "python uv pip install"},
        env_overrides={"AGENT_MEM_HOME": str(short_home)},
    )
    assert res.returncode == 0, res.stderr
    stdout = res.stdout.strip()
    assert stdout, "fallback should still produce additionalContext"
    output = json.loads(stdout)
    ctx = output["hookSpecificOutput"]["additionalContext"]
    assert "## Ultan — your library says" in ctx
    assert "use-uv-not-pip" in ctx
