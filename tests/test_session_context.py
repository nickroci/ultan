"""Tests for the SessionStart boot-context builder (G5).

``ultan/_session_context.build_session_start_context`` rebuilds the legacy
``src/hooks/session_start.py`` injection against the daemon's ``knowledge/``
layout (no ``daily/`` log anymore): date + project, an ``index.md`` head
excerpt, and a "recent activity" recap drawn from ``knowledge/log.md`` (or, when
that's absent, the most-recently-modified entries).

Hermetic: every test points ``AGENT_MEM_HOME`` at a tmp dir so nothing touches
the real store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ultan import _session_context


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "agent-mem"
    (h / "knowledge").mkdir(parents=True)
    monkeypatch.setenv("AGENT_MEM_HOME", str(h))
    return h


def _seed_index(home: Path, rows: int = 3) -> None:
    lines = [
        "# Knowledge Index",
        "",
        "| Article | Scope | Summary |",
        "|---|---|---|",
    ]
    for i in range(rows):
        lines.append(f"| [[global/concepts/entry-{i}]] | global | summary number {i} |")
    (home / "knowledge" / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_log(home: Path, entries: int = 3) -> None:
    blocks = ["# Scholar Action Log", ""]
    for i in range(entries):
        # Newest-first, matching the live log.md ordering.
        blocks.append(f"## [2026-06-{12 - i:02d}T10:00:00+00:00] compile | source-{i}.md")
        blocks.append(f"- Articles created: [[global/concepts/entry-{i}]]")
        blocks.append("")
    (home / "knowledge" / "log.md").write_text("\n".join(blocks) + "\n", encoding="utf-8")


# ── happy path: date + slug + index excerpt + activity ───────────────────────


def test_includes_date_slug_index_and_activity(home: Path, tmp_path: Path) -> None:
    _seed_index(home)
    _seed_log(home)
    # A non-git cwd → slug == basename.
    cwd = tmp_path / "my-project"
    cwd.mkdir()

    ctx = _session_context.build_session_start_context(str(cwd))

    # Date: the current year appears in the "Today" line.
    from datetime import datetime, timezone

    year = str(datetime.now(timezone.utc).astimezone().year)
    assert "## Today" in ctx
    assert year in ctx

    # Project slug derived from the cwd basename.
    assert "## Current project" in ctx
    assert "`my-project`" in ctx

    # Index excerpt present.
    assert "## Knowledge Base Index" in ctx
    assert "Knowledge Index" in ctx
    assert "entry-0" in ctx

    # Recent activity drawn from log.md (newest entry shows).
    assert "## Recent Library Activity" in ctx
    assert "compile | source-0.md" in ctx


# ── char cap is respected ────────────────────────────────────────────────────


def test_respects_char_cap(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink the cap so we can prove truncation without a giant fixture.
    monkeypatch.setattr(_session_context, "MAX_CONTEXT_CHARS", 200)
    # Oversized index so the assembled block would blow the cap.
    big = ["# Knowledge Index", ""] + [f"| [[e-{i}]] | global | {'x' * 80} |" for i in range(200)]
    (home / "knowledge" / "index.md").write_text("\n".join(big), encoding="utf-8")
    _seed_log(home)

    ctx = _session_context.build_session_start_context(str(tmp_path))

    assert len(ctx) <= 200 + len("\n…(truncated)")
    assert ctx.endswith("…(truncated)")


def test_index_section_is_head_excerpt_not_whole_file(home: Path, tmp_path: Path) -> None:
    # Far more rows than the per-section line cap → only the head should appear.
    rows = _session_context._MAX_INDEX_LINES + 50
    lines = ["# Knowledge Index", ""] + [f"| row-{i} | global | s |" for i in range(rows)]
    (home / "knowledge" / "index.md").write_text("\n".join(lines), encoding="utf-8")

    ctx = _session_context.build_session_start_context(str(tmp_path))

    assert "row-0" in ctx
    assert f"row-{rows - 1}" not in ctx  # tail rows are excluded
    assert "index truncated" in ctx


# ── recent activity: only the newest N log entries ───────────────────────────


def test_recent_activity_keeps_only_newest_log_entries(home: Path, tmp_path: Path) -> None:
    _seed_index(home)
    _seed_log(home, entries=_session_context._MAX_LOG_ENTRIES + 4)

    ctx = _session_context.build_session_start_context(str(tmp_path))

    assert "compile | source-0.md" in ctx  # newest kept
    # The (N+1)th-newest entry must be dropped (the cut is on the block boundary).
    dropped = f"compile | source-{_session_context._MAX_LOG_ENTRIES}.md"
    assert dropped not in ctx


# ── fallback: no log.md → recently-modified entries ──────────────────────────


def test_activity_falls_back_to_recent_entries_without_log(home: Path, tmp_path: Path) -> None:
    _seed_index(home)
    # No log.md. Seed a couple of real entries.
    gdir = home / "knowledge" / "global" / "concepts"
    gdir.mkdir(parents=True)
    (gdir / "alpha.md").write_text("# alpha\n", encoding="utf-8")
    (gdir / "beta.md").write_text("# beta\n", encoding="utf-8")
    # README is a catalog and must be excluded from the recent list.
    (home / "knowledge" / "README.md").write_text("# readme\n", encoding="utf-8")

    ctx = _session_context.build_session_start_context(str(tmp_path))

    assert "## Recent Library Activity" in ctx
    assert "Most recently updated entries" in ctx
    assert "global/concepts/alpha.md" in ctx
    assert "global/concepts/beta.md" in ctx
    assert "README.md" not in ctx  # catalog skipped


# ── empty store degrades to "" ───────────────────────────────────────────────


def test_empty_store_returns_empty_string(home: Path, tmp_path: Path) -> None:
    # knowledge/ exists (from the fixture) but has no index, no log, no entries.
    ctx = _session_context.build_session_start_context(str(tmp_path))
    assert ctx == ""


def test_missing_knowledge_dir_returns_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No knowledge/ at all.
    h = tmp_path / "bare-home"
    h.mkdir()
    monkeypatch.setenv("AGENT_MEM_HOME", str(h))
    assert _session_context.build_session_start_context(str(tmp_path)) == ""


# ── bucket resolution: existing bucket dir surfaces in the header ────────────


def test_header_names_matching_project_bucket(home: Path, tmp_path: Path) -> None:
    _seed_index(home)
    # A bucket dir whose name matches the cwd-derived slug.
    (home / "knowledge" / "projects" / "my-project").mkdir(parents=True)
    cwd = tmp_path / "my-project"
    cwd.mkdir()

    ctx = _session_context.build_session_start_context(str(cwd))

    assert "knowledge/projects/my-project/" in ctx


def test_no_bucket_still_notes_global_applies(home: Path, tmp_path: Path) -> None:
    _seed_index(home)
    ctx = _session_context.build_session_start_context(str(tmp_path))
    # No projects/ bucket on disk → step-2 candidate is the cwd basename; the
    # header always points at global/ as a floor.
    assert "knowledge/global/" in ctx


# ── never raises (cwd=None falls back to process cwd) ────────────────────────


def test_cwd_none_does_not_crash(home: Path) -> None:
    _seed_index(home)
    # Must not raise even with no cwd from the payload.
    ctx = _session_context.build_session_start_context(None)
    assert "## Today" in ctx
