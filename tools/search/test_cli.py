"""Integration tests driving the ``agent-mem`` CLI ``main()`` end-to-end.

Companion to ``test_cli_lifecycle.py``. That file exercises the lifecycle
command handlers directly (``cmd_promote``, ``cmd_demote`` …). This file
drives ``main()`` with crafted argv against a real ``tmp_path`` knowledge
dir, so the argparse wiring, hierarchy mode, BM25 mode, index mode (with
the SDK monkeypatched away), merged mode, and doctor-print path all get
covered without a network call.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date
from pathlib import Path

import pytest

import cli

# ── Knowledge-dir resolution ───────────────────────────────────────────────────


def test_resolve_knowledge_dir_prefers_explicit_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MEM_KNOWLEDGE", "/should-be-ignored")
    monkeypatch.setenv("AGENT_MEM_HOME", "/should-also-be-ignored")
    target = tmp_path / "explicit"
    target.mkdir()
    assert cli._resolve_knowledge_dir(str(target)) == target.resolve()


def test_resolve_knowledge_dir_uses_env_when_no_flag(tmp_path: Path, monkeypatch) -> None:
    env_target = tmp_path / "env-knowledge"
    env_target.mkdir()
    monkeypatch.setenv("AGENT_MEM_KNOWLEDGE", str(env_target))
    monkeypatch.delenv("AGENT_MEM_HOME", raising=False)
    assert cli._resolve_knowledge_dir(None) == env_target.resolve()


def test_resolve_knowledge_dir_falls_back_to_home(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "myhome"
    home.mkdir()
    monkeypatch.delenv("AGENT_MEM_KNOWLEDGE", raising=False)
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    assert cli._resolve_knowledge_dir(None) == (home / "knowledge").resolve()


def test_resolve_knowledge_dir_default_when_nothing_set(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MEM_KNOWLEDGE", raising=False)
    monkeypatch.delenv("AGENT_MEM_HOME", raising=False)
    assert cli._resolve_knowledge_dir(None) == cli.DEFAULT_KNOWLEDGE_DIR


def test_resolve_agent_mem_home_prefers_env(tmp_path: Path, monkeypatch) -> None:
    explicit_home = tmp_path / "explicit-home"
    monkeypatch.setenv("AGENT_MEM_HOME", str(explicit_home))
    # knowledge_dir is unrelated when env is set.
    assert cli._resolve_agent_mem_home(tmp_path / "ignored") == explicit_home.resolve()


def test_resolve_agent_mem_home_falls_back_to_knowledge_parent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MEM_HOME", raising=False)
    knowledge = tmp_path / "myhome" / "knowledge"
    knowledge.mkdir(parents=True)
    assert cli._resolve_agent_mem_home(knowledge) == tmp_path / "myhome"


# ── Hierarchy mode (function-level + via main) ─────────────────────────────────


def test_hierarchy_mode_missing_knowledge_dir(tmp_path: Path, capsys) -> None:
    rc = cli.hierarchy_mode(tmp_path / "nope", None)
    assert rc == 2
    assert "knowledge dir not found" in capsys.readouterr().err


def test_hierarchy_mode_lists_md_files(knowledge_dir: Path, capsys) -> None:
    rc = cli.hierarchy_mode(knowledge_dir, None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "factory-pattern-for-apis.md" in out
    assert "no-mock-db.md" in out
    # _archive excluded — but there isn't one in the fixture, just sanity check
    # nothing escaped the dir.
    for line in out.strip().splitlines():
        assert "_archive" not in line


def test_hierarchy_mode_subpath_with_no_md_prints_empty_message(
    knowledge_dir: Path, capsys
) -> None:
    empty_sub = knowledge_dir / "global" / "empty-dir"
    empty_sub.mkdir(parents=True)
    rc = cli.hierarchy_mode(knowledge_dir, "global/empty-dir")
    assert rc == 0
    assert "(no markdown entries under" in capsys.readouterr().out


def test_hierarchy_mode_subpath_pointing_at_file_prints_just_that(
    knowledge_dir: Path, capsys
) -> None:
    rc = cli.hierarchy_mode(knowledge_dir, "global/concepts/factory-pattern-for-apis.md")
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("factory-pattern-for-apis.md")


def test_hierarchy_mode_refuses_to_escape_knowledge_dir(knowledge_dir: Path, capsys) -> None:
    rc = cli.hierarchy_mode(knowledge_dir, "../../etc")
    assert rc == 2
    assert "refusing to walk outside" in capsys.readouterr().err


def test_hierarchy_mode_subpath_does_not_exist(knowledge_dir: Path, capsys) -> None:
    rc = cli.hierarchy_mode(knowledge_dir, "global/concepts/does-not-exist")
    assert rc == 1
    assert "path not found" in capsys.readouterr().err


def test_hierarchy_mode_skips_archive(knowledge_dir: Path, capsys) -> None:
    archived = knowledge_dir / "_archive" / "old.md"
    archived.parent.mkdir(parents=True)
    archived.write_text("# archived\n", encoding="utf-8")
    rc = cli.hierarchy_mode(knowledge_dir, None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "_archive" not in out


# ── BM25 mode helper ───────────────────────────────────────────────────────────


def test_bm25_mode_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert cli.bm25_mode(tmp_path / "nope", "factory pattern") == []


def test_bm25_mode_returns_hits(knowledge_dir: Path) -> None:
    hits = cli.bm25_mode(knowledge_dir, "factory pattern for APIs", k=5)
    assert hits
    assert all(h.sources == ["bm25"] for h in hits)


# ── Index mode: patched via the fake_sdk fixture (see conftest.py) ─────────────


def test_run_index_query_missing_index_file(tmp_path: Path, fake_sdk) -> None:
    """No ``index.md`` -> friendly message, empty citations."""
    fake_sdk["chunks"] = ["unused"]
    empty = tmp_path / "knowledge"
    empty.mkdir()
    answer, cited = asyncio.run(cli._run_index_query(empty, "anything"))
    assert "does not exist" in answer
    assert cited == []


def test_run_index_query_collects_sources(
    knowledge_dir: Path,
    fake_sdk,
) -> None:
    factory = knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md"
    nomock = knowledge_dir / "global" / "concepts" / "no-mock-db.md"
    answer_text = (
        "Here is an answer.\n"
        f"SOURCE: {factory}\n"
        f"SOURCE: {nomock}\n"
        f"SOURCE: {factory}\n"  # duplicate — should dedupe
        "SOURCE: \n"  # empty — should skip
        "SOURCE: /tmp/does/not/exist/anywhere.md\n"  # non-existent — skip
    )
    fake_sdk["chunks"] = [answer_text]
    answer, cited = asyncio.run(cli._run_index_query(knowledge_dir, "factory"))
    assert "Here is an answer." in answer
    # Dedupe + skip empty/non-existent.
    assert cited == [factory.resolve(), nomock.resolve()]


def test_run_index_query_handles_sdk_exception(
    knowledge_dir: Path,
    fake_sdk,
) -> None:
    fake_sdk["raise_exc"] = RuntimeError("boom")
    answer, cited = asyncio.run(cli._run_index_query(knowledge_dir, "x"))
    assert "[index mode error:" in answer
    assert "boom" in answer
    assert cited == []


def test_index_mode_sync_wrapper(knowledge_dir: Path, fake_sdk) -> None:
    """``index_mode`` is the sync wrapper used by the CLI."""
    factory = knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md"
    fake_sdk["chunks"] = [f"answer here\nSOURCE: {factory}\n"]
    answer, hits = cli.index_mode(knowledge_dir, "factory")
    assert "answer here" in answer
    assert len(hits) == 1
    assert hits[0].sources == ["index"]
    assert hits[0].path == factory.resolve()


# ── Merged mode ────────────────────────────────────────────────────────────────


def test_merged_mode_dedupes_overlapping_sources(
    knowledge_dir: Path,
    fake_sdk,
) -> None:
    """Both BM25 and index-led can surface the same file. The merge keeps
    one record with both sources tagged."""
    factory = knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md"
    fake_sdk["chunks"] = [f"x\nSOURCE: {factory}\n"]
    hits, answer = cli.merged_mode(knowledge_dir, "factory pattern for APIs", k=5)
    assert hits
    # factory must appear and carry both sources.
    factory_hit = next(h for h in hits if h.path.resolve() == factory.resolve())
    assert set(factory_hit.sources) == {"bm25", "index"}
    # Files only surfaced by one engine should still be present.
    assert any(set(h.sources) == {"bm25"} or set(h.sources) == {"index"} for h in hits)
    assert "x" in answer


def test_merged_mode_index_only_hits_preserved(
    knowledge_dir: Path,
    fake_sdk,
) -> None:
    """A path returned only by the index-led engine still gets a hit row."""
    # Use a query no BM25 token matches in the corpus.
    factory = knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md"
    fake_sdk["chunks"] = [f"only-index\nSOURCE: {factory}\n"]
    hits, _ = cli.merged_mode(knowledge_dir, "zzzzzzz qqqqqqq", k=5)
    factory_hit = next(h for h in hits if h.path.resolve() == factory.resolve())
    assert factory_hit.sources == ["index"]


# ── Hit printer ────────────────────────────────────────────────────────────────


def test_print_hits_handles_empty(capsys) -> None:
    cli._print_hits([], header="No hits here")
    out = capsys.readouterr().out
    assert "No hits here" in out
    assert "(no hits)" in out


def test_print_hits_renders_score_path_snippet(capsys) -> None:
    hit = cli.Hit(path=Path("/x/y.md"), score=4.2, snippet="snip", sources=["bm25"])
    cli._print_hits([hit], header="Hits:")
    out = capsys.readouterr().out
    assert "Hits:" in out
    assert "/x/y.md" in out
    assert "snip" in out
    assert "bm25" in out
    assert "4.20" in out


# ── _resolve_identifier ────────────────────────────────────────────────────────


def test_resolve_identifier_ambiguous_id(knowledge_dir: Path, make_provisional) -> None:
    """Two entries with the same ``id`` -> LookupError lists both."""
    make_provisional(knowledge_dir / "global" / "concepts" / "dupe-a.md", ident="duplicate-id")
    make_provisional(
        knowledge_dir / "projects" / "example-app" / "concepts" / "dupe-b.md",
        ident="duplicate-id",
        scope="project:example-app",
    )
    with pytest.raises(LookupError) as exc_info:
        cli._resolve_identifier(knowledge_dir, "duplicate-id")
    assert "ambiguous" in str(exc_info.value)
    assert "dupe-a.md" in str(exc_info.value)
    assert "dupe-b.md" in str(exc_info.value)


def test_resolve_identifier_absolute_path(knowledge_dir: Path) -> None:
    p = knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md"
    assert cli._resolve_identifier(knowledge_dir, str(p)) == p.resolve()
    # Without .md suffix too.
    assert cli._resolve_identifier(knowledge_dir, str(p.with_suffix(""))) == p.resolve()


def test_resolve_identifier_relative_to_cwd(knowledge_dir: Path, monkeypatch) -> None:
    """Path relative to cwd is one of the candidates we try."""
    monkeypatch.chdir(knowledge_dir / "global" / "concepts")
    assert (
        cli._resolve_identifier(knowledge_dir, "factory-pattern-for-apis.md")
        == (knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md").resolve()
    )


def test_short_id_falls_back_to_relative_path(tmp_path: Path) -> None:
    """When frontmatter has no ``id``, ``_short_id`` returns the relative path."""
    knowledge = tmp_path / "knowledge"
    p = knowledge / "no-id.md"
    p.parent.mkdir(parents=True)
    p.write_text("---\ntype: lesson\n---\nbody\n", encoding="utf-8")
    out = cli._short_id(knowledge, p)
    assert out == "no-id.md"


def test_short_id_outside_knowledge_dir(tmp_path: Path) -> None:
    """A path that isn't under the knowledge dir falls back to str(path)."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_text("---\ntype: lesson\n---\nbody\n", encoding="utf-8")
    assert cli._short_id(knowledge, elsewhere) == str(elsewhere)


# ── _safe_read_frontmatter ─────────────────────────────────────────────────────


def test_safe_read_frontmatter_returns_none_on_error(tmp_path: Path) -> None:
    """Broken YAML -> None (not an exception)."""
    p = tmp_path / "bad.md"
    p.write_text("---\nbad: : :\n---\nbody\n", encoding="utf-8")
    assert cli._safe_read_frontmatter(p) is None


def test_safe_read_frontmatter_returns_none_for_no_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "plain.md"
    p.write_text("just a body, no frontmatter\n", encoding="utf-8")
    # ``fm_read`` returns ({}, body) — ``_safe_read_frontmatter`` maps {} to None.
    assert cli._safe_read_frontmatter(p) is None


# ── cmd_forget edge cases ──────────────────────────────────────────────────────


def test_cmd_forget_refuses_path_outside_knowledge_dir(
    knowledge_dir: Path, tmp_path: Path, capsys
) -> None:
    """An identifier resolving to a path outside the knowledge dir is rejected."""
    outside = tmp_path / "outside.md"
    outside.write_text(
        "---\nid: outsider\ntype: lesson\nscope: global\nstatus: provisional\n---\nbody\n",
        encoding="utf-8",
    )
    rc = cli.cmd_forget(knowledge_dir, str(outside))
    assert rc == 2
    assert "not under knowledge dir" in capsys.readouterr().err


# ── Review interactive edge cases ──────────────────────────────────────────────


def test_review_unknown_action_loops(knowledge_dir: Path, make_provisional, capsys) -> None:
    """An unknown action should print a warning and re-prompt."""
    make_provisional(knowledge_dir / "global" / "concepts" / "rev-prov.md", ident="rev-prov")
    import io

    # First an unknown action, then quit.
    fake_stdin = io.StringIO("x\nq\n")
    rc = cli.cmd_review(knowledge_dir, noninteractive=False, stdin=fake_stdin)
    assert rc == 0
    err_out = capsys.readouterr().out
    assert "unknown action" in err_out


def test_review_edit_invokes_editor(knowledge_dir: Path, make_provisional, monkeypatch) -> None:
    """The 'e' action should fire ``_edit_in_editor`` and re-read the file."""
    target = make_provisional(
        knowledge_dir / "global" / "concepts" / "rev-prov.md", ident="rev-prov"
    )
    edits: list[Path] = []

    def fake_edit(p: Path) -> None:
        edits.append(p)

    monkeypatch.setattr(cli, "_edit_in_editor", fake_edit)

    import io

    # One 'e', then 'q' to exit.
    fake_stdin = io.StringIO("e\nq\n")
    rc = cli.cmd_review(knowledge_dir, noninteractive=False, stdin=fake_stdin)
    assert rc == 0
    assert any(p == target for p in edits)


def test_review_empty_input_quits_gracefully(knowledge_dir: Path, make_provisional) -> None:
    """Empty readline (EOF) should quit cleanly without crashing."""
    make_provisional(knowledge_dir / "global" / "concepts" / "rev-prov.md", ident="rev-prov")
    import io

    fake_stdin = io.StringIO("")  # immediate EOF
    rc = cli.cmd_review(knowledge_dir, noninteractive=False, stdin=fake_stdin)
    assert rc == 0


def test_edit_in_editor_invokes_subprocess(tmp_path: Path, monkeypatch) -> None:
    """``_edit_in_editor`` shells out to ``$EDITOR`` (or ``vi``)."""
    called: list[list[str]] = []

    def fake_call(cmd, *_a, **_kw):
        called.append(cmd)
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    monkeypatch.setenv("EDITOR", "myedit")
    target = tmp_path / "x.md"
    target.write_text("body\n", encoding="utf-8")
    cli._edit_in_editor(target)
    assert called == [["myedit", str(target)]]


def test_entry_preview_handles_applies_when_list(tmp_path: Path) -> None:
    """``_entry_preview`` walks both list and string ``applies-when`` formats."""
    p = tmp_path / "entry.md"
    p.write_text(
        "---\n"
        "id: x\n"
        "type: lesson\n"
        "scope: global\n"
        "status: provisional\n"
        "applies-when:\n"
        "  - foo\n"
        "  - bar\n"
        "---\n"
        "body line 1\n"
        "body line 2\n",
        encoding="utf-8",
    )
    out = cli._entry_preview(p)
    assert "applies-when: foo" in out
    assert "applies-when: bar" in out


# ── Doctor / run_doctor missing branches ───────────────────────────────────────


def test_pid_alive_for_self() -> None:
    """Our own PID is always alive."""
    assert cli._pid_alive(os.getpid()) is True


def test_pid_alive_handles_permission_error(monkeypatch) -> None:
    """When ``os.kill`` raises ``PermissionError``, the process is alive."""

    def fake_kill(_pid, _sig):
        raise PermissionError()

    monkeypatch.setattr(cli.os, "kill", fake_kill)
    assert cli._pid_alive(1) is True


def test_pid_alive_handles_oserror(monkeypatch) -> None:
    def fake_kill(_pid, _sig):
        raise OSError()

    monkeypatch.setattr(cli.os, "kill", fake_kill)
    assert cli._pid_alive(1) is False


def test_read_pid_file_missing_returns_none(tmp_path: Path) -> None:
    assert cli._read_pid_file(tmp_path / "no-such") is None


def test_read_pid_file_garbage_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "pid"
    p.write_text("not-a-number\n", encoding="utf-8")
    assert cli._read_pid_file(p) is None


def test_load_cost_state_missing_returns_none(tmp_path: Path) -> None:
    assert cli._load_cost_state(tmp_path / "no-cost.json") is None


def test_load_cost_state_corrupt_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "cost.json"
    p.write_text("not json {", encoding="utf-8")
    assert cli._load_cost_state(p) is None


def test_scope_label_handles_unset() -> None:
    assert cli._scope_label({}) == "(unset)"


def test_scope_label_handles_project_scope() -> None:
    assert cli._scope_label({"scope": "project:vol-predictor"}) == "project:vol-predictor"


def test_find_src_dir_returns_none_when_no_checkout(tmp_path: Path, monkeypatch) -> None:
    """Move the CLI's notional location to a tmp dir with no src — function
    should fall back to ``$AGENT_MEM_SRC`` and return ``None`` if that's also
    bogus."""
    monkeypatch.setenv("AGENT_MEM_SRC", str(tmp_path / "nope"))
    # Patch __file__ resolution so the parent walk lands somewhere harmless.
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "fake_cli.py"))
    assert cli._find_src_dir() is None


def test_find_src_dir_uses_agent_mem_src_env(tmp_path: Path, monkeypatch) -> None:
    fake_src = tmp_path / "src-checkout"
    (fake_src / "scripts").mkdir(parents=True)
    (fake_src / "scripts" / "lint.py").write_text("# noop\n", encoding="utf-8")
    # Make the auto-detect fail so we exercise the env fallback.
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "fake_cli.py"))
    monkeypatch.setenv("AGENT_MEM_SRC", str(fake_src))
    assert cli._find_src_dir() == fake_src.resolve()


def test_run_structural_lint_skipped_when_no_src(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_find_src_dir", lambda: None)
    rc, output = cli._run_structural_lint(tmp_path / "knowledge")
    assert rc < 0
    assert "could not locate" in output


def test_run_structural_lint_skipped_when_script_missing(tmp_path: Path, monkeypatch) -> None:
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    monkeypatch.setattr(cli, "_find_src_dir", lambda: fake_src)
    rc, output = cli._run_structural_lint(tmp_path / "knowledge")
    assert rc < 0
    assert "not found" in output


def test_run_structural_lint_skipped_when_uv_missing(tmp_path: Path, monkeypatch) -> None:
    fake_src = tmp_path / "src"
    (fake_src / "scripts").mkdir(parents=True)
    (fake_src / "scripts" / "lint.py").write_text("# noop\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_find_src_dir", lambda: fake_src)

    def fake_run(*_a, **_kw):
        raise FileNotFoundError("uv not on PATH")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    rc, output = cli._run_structural_lint(tmp_path / "knowledge")
    assert rc < 0
    assert "uv" in output.lower()


def test_run_structural_lint_timeout(tmp_path: Path, monkeypatch) -> None:
    fake_src = tmp_path / "src"
    (fake_src / "scripts").mkdir(parents=True)
    (fake_src / "scripts" / "lint.py").write_text("# noop\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_find_src_dir", lambda: fake_src)
    import subprocess as sp

    def fake_run(*a, **kw):
        raise sp.TimeoutExpired(cmd="uv", timeout=1)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    rc, output = cli._run_structural_lint(tmp_path / "knowledge")
    assert rc < 0
    assert "timed out" in output


def test_run_structural_lint_returns_real_output(tmp_path: Path, monkeypatch) -> None:
    """When the subprocess runs cleanly, output and rc are returned."""
    fake_src = tmp_path / "src"
    (fake_src / "scripts").mkdir(parents=True)
    (fake_src / "scripts" / "lint.py").write_text("# noop\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_find_src_dir", lambda: fake_src)

    class _Result:
        returncode = 0
        stdout = "lint ok\n"
        stderr = ""

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: _Result())
    rc, output = cli._run_structural_lint(tmp_path / "knowledge")
    assert rc == 0
    assert "lint ok" in output


def test_run_doctor_cost_rolled_over(home_and_knowledge: tuple[Path, Path], monkeypatch) -> None:
    """A ``cost.json`` from a previous day -> ``cost_today`` is forced to 0."""
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    (home / "cost.json").write_text(
        json.dumps({"today": "1999-01-01", "today_usd": 9.99, "lifetime_usd": 50.0}),
        encoding="utf-8",
    )
    # Skip lint subprocess for speed/determinism.
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (-1, "(skipped)"))
    report = cli.run_doctor(knowledge)
    assert report.cost_today == 0.0
    assert report.cost_lifetime == pytest.approx(50.0)


def test_run_doctor_pending_nudges_unreadable(
    home_and_knowledge: tuple[Path, Path], monkeypatch
) -> None:
    """An OSError reading pending-nudges.md surfaces as ``-1``."""
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    nudges = home / "pending-nudges.md"
    nudges.write_text("- nudge\n", encoding="utf-8")

    real_read_text = Path.read_text

    def fake_read_text(self, *a, **kw):
        if self == nudges:
            raise OSError("simulated")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (-1, "(skipped)"))
    report = cli.run_doctor(knowledge)
    assert report.pending_nudges == -1


def test_run_doctor_flags_lint_failure(home_and_knowledge: tuple[Path, Path], monkeypatch) -> None:
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (3, "issues found"))
    report = cli.run_doctor(knowledge)
    assert report.lint_rc == 3
    assert any("lint reported issues" in i for i in report.issues)


def test_run_doctor_running_daemon(home_and_knowledge: tuple[Path, Path], monkeypatch) -> None:
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    # Use our own PID — guaranteed alive.
    (home / "daemon.pid").write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (-1, "(skipped)"))
    report = cli.run_doctor(knowledge)
    assert report.daemon_status == "running"
    assert report.daemon_pid == os.getpid()


# ── cmd_doctor: print path ─────────────────────────────────────────────────────


def test_cmd_doctor_prints_full_report(
    home_and_knowledge: tuple[Path, Path], monkeypatch, capsys
) -> None:
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    today_iso = date.today().isoformat()
    (home / "cost.json").write_text(
        json.dumps({"today": today_iso, "today_usd": 0.5, "lifetime_usd": 1.5}),
        encoding="utf-8",
    )
    (home / "pending-nudges.md").write_text("# Nudges\n\n- one\n- two\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (0, "all good"))

    rc = cli.cmd_doctor(knowledge)
    out = capsys.readouterr().out
    assert rc == 0
    assert "agent-mem doctor" in out
    assert "Corpus:" in out
    assert "by status:" in out
    assert "by scope:" in out
    assert "Daemon:" in out
    assert "Cost:" in out
    assert "Pending nudges:" in out
    assert "Lint (structural-only):" in out
    assert "All checks clean." in out


def test_cmd_doctor_returns_nonzero_when_issues(
    home_and_knowledge: tuple[Path, Path], monkeypatch, capsys
) -> None:
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (1, "broken"))
    rc = cli.cmd_doctor(knowledge)
    assert rc == 1
    out = capsys.readouterr().out
    assert "Issues:" in out
    assert "lint reported issues" in out


def test_cmd_doctor_marks_stale_daemon(
    home_and_knowledge: tuple[Path, Path], monkeypatch, capsys
) -> None:
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    (home / "daemon.pid").write_text("999999", encoding="utf-8")
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (-1, "(skipped)"))
    rc = cli.cmd_doctor(knowledge)
    out = capsys.readouterr().out
    assert "[!!]" in out
    assert rc == 1


def test_cmd_doctor_marks_running_daemon(
    home_and_knowledge: tuple[Path, Path], monkeypatch, capsys
) -> None:
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    (home / "daemon.pid").write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (-1, "(skipped)"))
    rc = cli.cmd_doctor(knowledge)
    out = capsys.readouterr().out
    assert "[OK]" in out
    assert rc == 0


def test_cmd_doctor_handles_singular_entry(tmp_path: Path, monkeypatch, capsys) -> None:
    """Edge case: 1 entry — the plural/singular path is hit."""
    home = tmp_path / "home"
    knowledge = home / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "only.md").write_text(
        "---\nid: only\ntype: lesson\nscope: global\nstatus: confirmed\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (-1, "(skipped)"))
    rc = cli.cmd_doctor(knowledge)
    out = capsys.readouterr().out
    assert "1 entry" in out
    assert rc == 0


# ── main(): argv-driven integration ───────────────────────────────────────────


def test_main_no_command_errors() -> None:
    """argparse exits with code 2 when no subcommand is given."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_main_search_hierarchy(knowledge_dir: Path, monkeypatch, capsys) -> None:
    rc = cli.main(["--knowledge-dir", str(knowledge_dir), "search", "--hierarchy"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "factory-pattern-for-apis.md" in out


def test_main_search_hierarchy_with_subpath(knowledge_dir: Path, capsys) -> None:
    rc = cli.main(
        [
            "--knowledge-dir",
            str(knowledge_dir),
            "search",
            "--hierarchy",
            "global/concepts",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "factory-pattern-for-apis.md" in out
    # projects subtree must NOT be in the output.
    assert "example-app" not in out


def test_main_search_requires_query_without_hierarchy(
    knowledge_dir: Path,
) -> None:
    """argparse `parser.error` calls SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--knowledge-dir", str(knowledge_dir), "search"])
    assert exc.value.code == 2


def test_main_search_bm25_mode(knowledge_dir: Path, capsys) -> None:
    rc = cli.main(
        [
            "--knowledge-dir",
            str(knowledge_dir),
            "search",
            "--bm25",
            "factory",
            "pattern",
            "-k",
            "3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "BM25 results" in out
    assert "factory-pattern-for-apis.md" in out


def test_main_search_bm25_no_hits_prints_hint(knowledge_dir: Path, capsys) -> None:
    rc = cli.main(
        [
            "--knowledge-dir",
            str(knowledge_dir),
            "search",
            "--bm25",
            "zzzzzzz",
            "qqqqqq",
        ]
    )
    assert rc == 0
    assert "try --index" in capsys.readouterr().out


def test_main_search_index_mode(knowledge_dir: Path, capsys, fake_sdk) -> None:
    factory = knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md"
    fake_sdk["chunks"] = [f"my answer\nSOURCE: {factory}\n"]
    rc = cli.main(
        [
            "--knowledge-dir",
            str(knowledge_dir),
            "search",
            "--index",
            "factory",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Index-led answer" in out
    assert "my answer" in out
    assert "Entries cited" in out


def test_main_search_merged_default(knowledge_dir: Path, capsys, fake_sdk) -> None:
    factory = knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md"
    fake_sdk["chunks"] = [f"synth answer\nSOURCE: {factory}\n"]
    rc = cli.main(
        [
            "--knowledge-dir",
            str(knowledge_dir),
            "search",
            "factory",
            "pattern",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Merged search" in out
    assert "Entries:" in out
    assert "Index-led synthesis:" in out
    assert "synth answer" in out


def test_main_search_merged_no_results_prints_advice(tmp_path: Path, monkeypatch, capsys) -> None:
    """Empty knowledge dir + SDK returns no useful answer — main() prints
    the friendly "nothing matched" hint."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    # No index.md present -> _run_index_query returns "[index mode: ... does not exist..."
    rc = cli.main(
        [
            "--knowledge-dir",
            str(knowledge),
            "search",
            "nothing",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Nothing matched" in out


def test_main_promote_via_argv(knowledge_dir: Path, capsys, make_provisional) -> None:
    make_provisional(knowledge_dir / "global" / "concepts" / "argv-prov.md", ident="argv-prov")
    rc = cli.main(["--knowledge-dir", str(knowledge_dir), "promote", "argv-prov"])
    assert rc == 0
    assert "promoted" in capsys.readouterr().out


def test_main_demote_via_argv(knowledge_dir: Path, capsys) -> None:
    rc = cli.main(
        [
            "--knowledge-dir",
            str(knowledge_dir),
            "demote",
            "factory-pattern-for-apis",
        ]
    )
    assert rc == 0
    assert "demoted" in capsys.readouterr().out


def test_main_forget_via_argv(knowledge_dir: Path, make_provisional, capsys) -> None:
    make_provisional(knowledge_dir / "global" / "concepts" / "tmp.md", ident="tmp-forget")
    rc = cli.main(["--knowledge-dir", str(knowledge_dir), "forget", "tmp-forget"])
    assert rc == 0
    assert "archived" in capsys.readouterr().out


def test_main_review_noninteractive_via_argv(knowledge_dir: Path, capsys) -> None:
    rc = cli.main(["--knowledge-dir", str(knowledge_dir), "review", "--noninteractive"])
    assert rc == 0
    out = capsys.readouterr().out
    # fixture has at least one provisional entry (auth-redirects).
    assert "auth-redirects" in out or "would be reviewed" in out


def test_main_doctor_via_argv(home_and_knowledge: tuple[Path, Path], monkeypatch, capsys) -> None:
    home, knowledge = home_and_knowledge
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.setattr(cli, "_run_structural_lint", lambda kd: (-1, "(skipped)"))
    rc = cli.main(["--knowledge-dir", str(knowledge), "doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "agent-mem doctor" in out
