"""Tests for the memory decay engine.

Coverage:
  - frontmatter mutation (``stamp_last_surfaced``): preserves other
    fields, atomic against partial writes, idempotent within a day.
  - sweep-state I/O: roundtrip plus parse-error fallback.
  - eligibility predicate: each of the three archival conditions
    (age, reinforced floor, arousal pin) and their interactions.
  - end-to-end ``run_sweep``: archives only eligible entries, leaves
    the rest untouched, writes sweep-state on completion.
  - trigger guard: ``maybe_run_sweep`` self-skips inside the cooldown
    and inside the lock.

Tests use ``today`` injection on the public functions so they're
deterministic against the system clock.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from agent_mem_daemon import decay

# ── Fixtures ─────────────────────────────────────────────────────────


def _write_entry(
    path: Path,
    *,
    id_: str,
    title: str,
    body: str = "A short body for testing.",
    reinforced: int = 0,
    arousal_pin: bool = False,
    last_surfaced: str | None = None,
    last_reinforced: str | None = None,
    created: str = "2026-01-01",
    updated: str | None = None,
    status: str | None = None,
) -> None:
    """Create an entry with the supplied frontmatter fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fm: Dict[str, Any] = {
        "id": id_,
        "title": title,
        "created": created,
        "reinforced": reinforced,
    }
    if updated is not None:
        fm["updated"] = updated
    if last_surfaced is not None:
        fm["last_surfaced"] = last_surfaced
    if last_reinforced is not None:
        fm["last_reinforced"] = last_reinforced
    if arousal_pin:
        fm["arousal_pin"] = True
    if status is not None:
        fm["status"] = status
    fm_text = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False).rstrip()
    path.write_text(f"---\n{fm_text}\n---\n\n# {title}\n\n{body}\n", encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated AGENT_MEM_HOME for each test."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    (tmp_path / "knowledge").mkdir(parents=True)
    return tmp_path


# ── stamp_last_surfaced ──────────────────────────────────────────────


def test_stamp_last_surfaced_writes_the_field(home: Path) -> None:
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo")
    ok = decay.stamp_last_surfaced(p, today_iso="2026-05-27")
    assert ok is True
    text = p.read_text(encoding="utf-8")
    assert "last_surfaced: '2026-05-27'" in text or "last_surfaced: 2026-05-27" in text


def test_stamp_last_surfaced_preserves_other_frontmatter(home: Path) -> None:
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(
        p,
        id_="foo",
        title="Foo",
        reinforced=3,
        last_reinforced="2026-04-01",
    )
    decay.stamp_last_surfaced(p, today_iso="2026-05-27")
    fm_block = p.read_text(encoding="utf-8").split("---")[1]
    fm = yaml.safe_load(fm_block)
    assert fm["id"] == "foo"
    assert fm["title"] == "Foo"
    assert fm["reinforced"] == 3
    assert fm["last_reinforced"] == "2026-04-01"
    assert str(fm["last_surfaced"]) == "2026-05-27"


def test_stamp_last_surfaced_is_idempotent_within_a_day(home: Path) -> None:
    """Calling twice the same day shouldn't churn the file mtime."""
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo")
    decay.stamp_last_surfaced(p, today_iso="2026-05-27")
    mtime1 = p.stat().st_mtime
    # Second call same day — should be a no-op (no rewrite).
    decay.stamp_last_surfaced(p, today_iso="2026-05-27")
    mtime2 = p.stat().st_mtime
    assert mtime2 == mtime1


def test_stamp_last_surfaced_returns_false_for_missing_file(home: Path) -> None:
    p = home / "knowledge" / "global" / "nonexistent.md"
    assert decay.stamp_last_surfaced(p, today_iso="2026-05-27") is False


def test_stamp_last_surfaced_returns_false_for_no_frontmatter(home: Path) -> None:
    p = home / "knowledge" / "global" / "raw.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Just a heading\n\nNo frontmatter here.\n")
    assert decay.stamp_last_surfaced(p, today_iso="2026-05-27") is False


# ── sweep-state I/O ──────────────────────────────────────────────────


def test_sweep_state_roundtrip(home: Path) -> None:
    when = datetime(2026, 5, 27, 18, 30, tzinfo=timezone.utc)
    decay.write_last_sweep_at(when)
    got = decay.read_last_sweep_at()
    assert got is not None
    assert got == when


def test_sweep_state_missing_file_returns_none(home: Path) -> None:
    assert decay.read_last_sweep_at() is None


def test_sweep_state_corrupt_file_returns_none(home: Path) -> None:
    (home / "sweep-state.json").write_text("not json at all")
    assert decay.read_last_sweep_at() is None


# ── Eligibility predicate ────────────────────────────────────────────


def test_eligible_when_old_and_low_reinforced() -> None:
    today = date(2026, 5, 27)
    fm = {"created": "2026-01-01", "reinforced": 0}
    assert decay._is_eligible_for_archive(fm, today=today) is True


def test_not_eligible_when_recent() -> None:
    today = date(2026, 5, 27)
    fm = {"created": "2026-05-20", "reinforced": 0}
    # 7 days < 30-day window.
    assert decay._is_eligible_for_archive(fm, today=today) is False


def test_not_eligible_when_reinforced_above_floor() -> None:
    today = date(2026, 5, 27)
    fm = {"created": "2026-01-01", "reinforced": 5}
    assert decay._is_eligible_for_archive(fm, today=today) is False


def test_not_eligible_when_arousal_pinned() -> None:
    today = date(2026, 5, 27)
    fm = {"created": "2026-01-01", "reinforced": 0, "arousal_pin": True}
    assert decay._is_eligible_for_archive(fm, today=today) is False


def test_already_stale_not_re_eligible() -> None:
    today = date(2026, 5, 27)
    fm = {"created": "2026-01-01", "reinforced": 0, "status": "stale"}
    assert decay._is_eligible_for_archive(fm, today=today) is False


def test_no_timestamps_conservatively_kept() -> None:
    """Missing all date fields → keep, don't archive on missing data."""
    today = date(2026, 5, 27)
    fm = {"reinforced": 0}
    assert decay._is_eligible_for_archive(fm, today=today) is False


def test_last_surfaced_extends_lifetime() -> None:
    """An entry created long ago but surfaced recently should survive —
    surfacing is the new activity signal the sweep respects."""
    today = date(2026, 5, 27)
    fm = {
        "created": "2026-01-01",
        "last_surfaced": "2026-05-20",  # 7 days ago
        "reinforced": 0,
    }
    assert decay._is_eligible_for_archive(fm, today=today) is False


# ── run_sweep end-to-end ─────────────────────────────────────────────


def test_run_sweep_archives_old_entries(home: Path) -> None:
    k = home / "knowledge"
    _write_entry(k / "global" / "old.md", id_="old", title="Old", created="2026-01-01")
    _write_entry(k / "global" / "recent.md", id_="recent", title="Recent", created="2026-05-20")
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)

    result = decay.run_sweep(knowledge_dir=k, now=when)

    assert result.archived == 1
    assert result.kept == 1
    assert result.errored == 0
    # Old entry moved to _archive preserving structure.
    archived_path = k / "_archive" / "global" / "old.md"
    assert archived_path.exists()
    assert not (k / "global" / "old.md").exists()
    # Recent entry untouched.
    assert (k / "global" / "recent.md").exists()


def test_run_sweep_stamps_archived_status_and_date(home: Path) -> None:
    k = home / "knowledge"
    _write_entry(k / "global" / "old.md", id_="old", title="Old", created="2026-01-01")
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)

    decay.run_sweep(knowledge_dir=k, now=when)
    archived = k / "_archive" / "global" / "old.md"
    fm_block = archived.read_text(encoding="utf-8").split("---")[1]
    fm = yaml.safe_load(fm_block)
    assert fm["status"] == "stale"
    assert str(fm["archived"]) == "2026-05-27"


def test_run_sweep_respects_reinforced_floor(home: Path) -> None:
    k = home / "knowledge"
    _write_entry(
        k / "global" / "old_but_reinforced.md",
        id_="old_but_reinforced",
        title="Old but reinforced",
        created="2026-01-01",
        reinforced=5,
    )
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)
    result = decay.run_sweep(knowledge_dir=k, now=when)
    assert result.archived == 0
    assert result.kept == 1


def test_run_sweep_respects_arousal_pin(home: Path) -> None:
    k = home / "knowledge"
    _write_entry(
        k / "global" / "pinned.md",
        id_="pinned",
        title="Pinned",
        created="2026-01-01",
        arousal_pin=True,
    )
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)
    result = decay.run_sweep(knowledge_dir=k, now=when)
    assert result.archived == 0


def test_run_sweep_writes_sweep_state(home: Path) -> None:
    k = home / "knowledge"
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)
    decay.run_sweep(knowledge_dir=k, now=when)
    assert decay.read_last_sweep_at() == when


def test_run_sweep_skips_archive_subtree(home: Path) -> None:
    """Already-archived entries don't get re-archived (no infinite churn)."""
    k = home / "knowledge"
    _write_entry(
        k / "_archive" / "global" / "old.md",
        id_="old",
        title="Already archived",
        created="2026-01-01",
        status="stale",
    )
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)
    result = decay.run_sweep(knowledge_dir=k, now=when)
    assert result.archived == 0
    # Still in the archive.
    assert (k / "_archive" / "global" / "old.md").exists()


# ── maybe_run_sweep / should_sweep cooldown ──────────────────────────


def test_should_sweep_true_when_never_run(home: Path) -> None:
    assert decay.should_sweep() is True


def test_should_sweep_false_within_cooldown(home: Path) -> None:
    when_last = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    decay.write_last_sweep_at(when_last)
    # 12h later — under the 24h cooldown.
    assert decay.should_sweep(now=when_last + timedelta(hours=12)) is False


def test_should_sweep_true_after_cooldown(home: Path) -> None:
    when_last = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    decay.write_last_sweep_at(when_last)
    assert decay.should_sweep(now=when_last + timedelta(hours=25)) is True


def test_maybe_run_sweep_skips_under_cooldown(home: Path) -> None:
    when_last = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    decay.write_last_sweep_at(when_last)
    out = decay.maybe_run_sweep(home / "knowledge", now=when_last + timedelta(hours=1))
    assert out is None


def test_maybe_run_sweep_runs_after_cooldown(home: Path) -> None:
    k = home / "knowledge"
    _write_entry(k / "global" / "old.md", id_="old", title="Old", created="2026-01-01")
    when_last = datetime(2026, 5, 1, tzinfo=timezone.utc)
    decay.write_last_sweep_at(when_last)
    when_now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    out = decay.maybe_run_sweep(k, now=when_now)
    assert out is not None
    assert out.archived == 1
