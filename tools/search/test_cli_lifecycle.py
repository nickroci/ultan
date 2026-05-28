"""Lifecycle subcommand tests: review / promote / demote / forget / doctor."""

from __future__ import annotations

import io
import shutil
from datetime import date
from pathlib import Path
from typing import Callable

import pytest

import cli
from frontmatter import read as fm_read

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def knowledge_dir(tmp_path: Path, make_provisional: Callable[..., Path]) -> Path:
    """A fresh copy of the fixture corpus + two provisional entries."""
    dst = tmp_path / "knowledge"
    shutil.copytree(FIXTURES, dst)
    make_provisional(dst / "global" / "concepts" / "prov-one.md", ident="prov-one")
    make_provisional(
        dst / "projects" / "example-app" / "concepts" / "prov-two.md",
        ident="prov-two",
        scope="project:example-app",
    )
    return dst


# ── promote ────────────────────────────────────────────────────────────────────


def test_promote_by_id_flips_status(knowledge_dir: Path) -> None:
    rc = cli.cmd_promote(knowledge_dir, "prov-one")
    assert rc == 0
    fm, _ = fm_read(knowledge_dir / "global" / "concepts" / "prov-one.md")
    assert fm["status"] == "confirmed"
    assert fm["updated"] == date.today().isoformat()


def test_promote_by_path_works(knowledge_dir: Path) -> None:
    rel = "projects/example-app/concepts/prov-two.md"
    rc = cli.cmd_promote(knowledge_dir, str(knowledge_dir / rel))
    assert rc == 0
    fm, _ = fm_read(knowledge_dir / rel)
    assert fm["status"] == "confirmed"


def test_promote_refuses_already_confirmed(knowledge_dir: Path, capsys) -> None:
    rc = cli.cmd_promote(knowledge_dir, "factory-pattern-for-apis")
    assert rc != 0
    err = capsys.readouterr().err
    assert "already confirmed" in err.lower()


def test_promote_unknown_id_errors(knowledge_dir: Path, capsys) -> None:
    rc = cli.cmd_promote(knowledge_dir, "does-not-exist-anywhere")
    assert rc == 2
    err = capsys.readouterr().err
    assert "no entry found" in err.lower()


# ── demote ─────────────────────────────────────────────────────────────────────


def test_demote_confirmed_to_provisional(knowledge_dir: Path) -> None:
    rc = cli.cmd_demote(knowledge_dir, "factory-pattern-for-apis")
    assert rc == 0
    fm, _ = fm_read(knowledge_dir / "global" / "concepts" / "factory-pattern-for-apis.md")
    assert fm["status"] == "provisional"


def test_demote_already_provisional_refuses(knowledge_dir: Path, capsys) -> None:
    rc = cli.cmd_demote(knowledge_dir, "prov-one")
    assert rc != 0
    assert "already provisional" in capsys.readouterr().err.lower()


# ── forget ─────────────────────────────────────────────────────────────────────


def test_forget_moves_file_to_archive(knowledge_dir: Path) -> None:
    src = knowledge_dir / "global" / "concepts" / "prov-one.md"
    assert src.exists()
    rc = cli.cmd_forget(knowledge_dir, "prov-one")
    assert rc == 0
    assert not src.exists()
    archived = knowledge_dir / "_archive" / "global" / "concepts" / "prov-one.md"
    assert archived.exists()


def test_forget_preserves_project_path(knowledge_dir: Path) -> None:
    rc = cli.cmd_forget(knowledge_dir, "prov-two")
    assert rc == 0
    archived = knowledge_dir / "_archive" / "projects" / "example-app" / "concepts" / "prov-two.md"
    assert archived.exists()


def test_forget_refuses_when_target_exists(knowledge_dir: Path, capsys) -> None:
    # Pre-create the archive target.
    target = knowledge_dir / "_archive" / "global" / "concepts" / "prov-one.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("already there\n", encoding="utf-8")

    rc = cli.cmd_forget(knowledge_dir, "prov-one")
    assert rc != 0
    err = capsys.readouterr().err
    assert "already exists" in err


# ── review ─────────────────────────────────────────────────────────────────────


def test_review_noninteractive_lists_provisional_only(knowledge_dir: Path, capsys) -> None:
    rc = cli.cmd_review(knowledge_dir, noninteractive=True)
    assert rc == 0
    out = capsys.readouterr().out
    # Both provisional entries appear (prov-one is global, auth-redirects is in the fixture).
    assert "prov-one" in out or "prov-one.md" in out
    assert "prov-two" in out or "prov-two.md" in out
    assert "auth-redirects" in out
    # Confirmed entries do not appear.
    assert "factory-pattern-for-apis" not in out
    assert "no-mock-db" not in out


def test_review_promotes_via_scripted_stdin(knowledge_dir: Path) -> None:
    """Answer 'p' to every prompt: every provisional entry becomes confirmed."""
    n_before = len(cli._list_provisional(knowledge_dir))
    assert n_before >= 2

    fake_stdin = io.StringIO("p\n" * (n_before + 5))  # extra newlines are harmless
    rc = cli.cmd_review(knowledge_dir, noninteractive=False, stdin=fake_stdin)
    assert rc == 0

    n_after = len(cli._list_provisional(knowledge_dir))
    assert n_after == 0


def test_review_rejects_via_scripted_stdin(knowledge_dir: Path) -> None:
    n_before = len(cli._list_provisional(knowledge_dir))
    assert n_before >= 2

    fake_stdin = io.StringIO("r\n" * (n_before + 5))
    rc = cli.cmd_review(knowledge_dir, noninteractive=False, stdin=fake_stdin)
    assert rc == 0

    # Every provisional entry should now be under _archive/.
    assert len(cli._list_provisional(knowledge_dir)) == 0
    archive = knowledge_dir / "_archive"
    archived_count = sum(1 for _ in archive.rglob("*.md"))
    assert archived_count >= n_before


def test_review_quit_stops_iteration(knowledge_dir: Path) -> None:
    n_before = len(cli._list_provisional(knowledge_dir))
    assert n_before >= 2
    # Quit immediately on the first entry.
    fake_stdin = io.StringIO("q\n")
    rc = cli.cmd_review(knowledge_dir, noninteractive=False, stdin=fake_stdin)
    assert rc == 0
    # Nothing should have changed.
    assert len(cli._list_provisional(knowledge_dir)) == n_before


def test_review_skip_leaves_entry_alone(knowledge_dir: Path) -> None:
    n_before = len(cli._list_provisional(knowledge_dir))
    fake_stdin = io.StringIO("s\n" * (n_before + 5))
    rc = cli.cmd_review(knowledge_dir, noninteractive=False, stdin=fake_stdin)
    assert rc == 0
    assert len(cli._list_provisional(knowledge_dir)) == n_before


def test_review_empty_corpus_returns_zero(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "knowledge"
    empty.mkdir()
    rc = cli.cmd_review(empty, noninteractive=False, stdin=io.StringIO(""))
    assert rc == 0
    assert "no provisional" in capsys.readouterr().out.lower()


# ── doctor ─────────────────────────────────────────────────────────────────────


def _setup_home(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal ~/.agent-mem/ home tree and return (home, knowledge_dir)."""
    home = tmp_path / "home"
    knowledge = home / "knowledge"
    shutil.copytree(FIXTURES, knowledge)
    return home, knowledge


def test_doctor_clean_fixture(tmp_path: Path, monkeypatch, capsys) -> None:
    home, knowledge = _setup_home(tmp_path)
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))

    report = cli.run_doctor(knowledge)
    assert report.daemon_status == "absent"
    assert report.daemon_pid is None
    assert report.cost_today == 0.0
    assert report.total_entries >= 5
    # Status counts include the fixture's confirmed + provisional entries.
    assert report.counts_by_status.get("confirmed", 0) >= 3
    assert report.counts_by_status.get("provisional", 0) >= 1
    # Scope mix: both `global` and `project:example-app` show up.
    assert "global" in report.counts_by_scope
    assert any(s.startswith("project:") for s in report.counts_by_scope)


def test_doctor_detects_stale_pid(tmp_path: Path, monkeypatch) -> None:
    home, knowledge = _setup_home(tmp_path)
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    # PID 1 is init/launchd — almost certainly alive AND we don't own it.
    # Use a high PID we're confident isn't allocated.
    (home / "daemon.pid").write_text("999999\n", encoding="utf-8")

    report = cli.run_doctor(knowledge)
    assert report.daemon_status == "stale"
    assert any("stale" in i.lower() for i in report.issues)


def test_doctor_reads_cost_file(tmp_path: Path, monkeypatch) -> None:
    home, knowledge = _setup_home(tmp_path)
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    today_iso = date.today().isoformat()
    (home / "cost.json").write_text(
        f'{{"today": "{today_iso}", "today_usd": 1.23, "lifetime_usd": 99.99}}',
        encoding="utf-8",
    )

    report = cli.run_doctor(knowledge)
    assert report.cost_today == pytest.approx(1.23)
    assert report.cost_lifetime == pytest.approx(99.99)
    # No more cost-cap field — enforcement was removed (tracking stays).
    assert not hasattr(report, "cost_cap")


def test_doctor_counts_pending_nudges(tmp_path: Path, monkeypatch) -> None:
    home, knowledge = _setup_home(tmp_path)
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    (home / "pending-nudges.md").write_text(
        "# Pending nudges\n"
        "\n"
        "- factory-pattern-for-apis: use factories\n"
        "- no-mock-db: real postgres\n",
        encoding="utf-8",
    )

    report = cli.run_doctor(knowledge)
    assert report.pending_nudges == 2


def test_doctor_dirty_fixture_broken_wikilink(tmp_path: Path, monkeypatch) -> None:
    """A broken wikilink should be visible in the lint output if lint is reachable.

    If lint can't be invoked (no `uv` on PATH, missing src checkout) the test
    degrades to checking that the doctor itself doesn't crash. We don't fail
    the suite on environment quirks.
    """
    home, knowledge = _setup_home(tmp_path)
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))

    # Inject a broken wikilink into an existing entry.
    broken = knowledge / "global" / "concepts" / "factory-pattern-for-apis.md"
    broken.write_text(
        broken.read_text(encoding="utf-8") + "\n\nSee [[global/concepts/does-not-exist]].\n",
        encoding="utf-8",
    )

    report = cli.run_doctor(knowledge)
    if report.lint_rc < 0:
        pytest.skip(f"lint unavailable: {report.lint_output!r}")
    # When lint runs, the broken link should surface as an error and the rc
    # should be non-zero.
    assert "broken" in report.lint_output.lower() or report.lint_rc > 0
