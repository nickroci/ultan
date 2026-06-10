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
    fired: int | None = None,
) -> None:
    """Create an entry with the supplied frontmatter fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fm: Dict[str, Any] = {
        "id": id_,
        "title": title,
        "created": created,
        "reinforced": reinforced,
    }
    if fired is not None:
        fm["fired"] = fired
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


# ── record_surface (fired counter; Defect 2) ─────────────────────────


def test_record_surface_increments_fired_and_stamps_last_surfaced(home: Path) -> None:
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo", fired=0)
    ok = decay.record_surface(p, today_iso="2026-05-27")
    assert ok is True
    fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    assert fm["fired"] == 1
    assert str(fm["last_surfaced"]) == "2026-05-27"


def test_record_surface_is_per_event_not_daily_gated(home: Path) -> None:
    """The crux of the invariant: two surface events on the SAME day must
    bump ``fired`` twice (unlike ``last_surfaced``, which is daily-idempotent)
    so ``fired-helpful`` — which can bump several times a day — never exceeds
    ``fired``."""
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo", fired=0)
    decay.record_surface(p, today_iso="2026-05-27")
    decay.record_surface(p, today_iso="2026-05-27")
    fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    assert fm["fired"] == 2
    assert str(fm["last_surfaced"]) == "2026-05-27"


def test_record_surface_starts_from_zero_when_field_absent(home: Path) -> None:
    """An entry with no ``fired`` field starts the counter at 1, not crash."""
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo")  # no fired field written
    assert "fired:" not in p.read_text(encoding="utf-8")
    decay.record_surface(p, today_iso="2026-05-27")
    fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    assert fm["fired"] == 1


def test_record_surface_coerces_bad_fired_value(home: Path) -> None:
    """A non-int ``fired`` (corrupt frontmatter) is treated as 0, then +1."""
    p = home / "knowledge" / "global" / "foo.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\nid: foo\ntitle: Foo\ncreated: '2026-01-01'\nfired: not-a-number\n---\n\nbody\n"
    )
    decay.record_surface(p, today_iso="2026-05-27")
    fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    assert fm["fired"] == 1


def test_record_surface_preserves_other_frontmatter(home: Path) -> None:
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo", fired=4, reinforced=3, last_reinforced="2026-04-01")
    decay.record_surface(p, today_iso="2026-05-27")
    fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    assert fm["id"] == "foo"
    assert fm["reinforced"] == 3
    assert fm["last_reinforced"] == "2026-04-01"
    assert fm["fired"] == 5


def test_record_surface_returns_false_for_missing_file(home: Path) -> None:
    p = home / "knowledge" / "global" / "nonexistent.md"
    assert decay.record_surface(p, today_iso="2026-05-27") is False


def test_record_surface_returns_false_for_no_frontmatter(home: Path) -> None:
    p = home / "knowledge" / "global" / "raw.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Just a heading\n\nNo frontmatter here.\n")
    assert decay.record_surface(p, today_iso="2026-05-27") is False


def test_fired_helpful_never_exceeds_fired_under_surface_semantics(home: Path) -> None:
    """Invariant guard: with N surface events and M<=N helpful bumps in the
    same day, ``fired-helpful <= fired`` holds. ``record_surface`` counts each
    surface; ``fired-helpful`` (modelled here as a same-day multi-bump) stays
    bounded by it."""
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo", fired=0)
    # Three surface events today.
    for _ in range(3):
        decay.record_surface(p, today_iso="2026-05-27")
    fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    fired = int(fm["fired"])
    # The Scholar would bump fired-helpful at most once per surfaced turn; the
    # denominator (fired) must dominate. Two helpful bumps <= three surfaces.
    fired_helpful = 2
    assert fired == 3
    assert fired_helpful <= fired


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


# ── Defensive paths (rarely-hit error branches) ──────────────────────


def test_run_sweep_uses_default_knowledge_dir_when_omitted(home: Path) -> None:
    """Calling without a kwarg should resolve to the env-pointed home."""
    # ``home`` fixture sets AGENT_MEM_HOME and creates knowledge/.
    result = decay.run_sweep()
    # Empty library → no archives, but the sweep should still write state.
    assert result.archived == 0
    assert decay.read_last_sweep_at() is not None


def test_run_sweep_missing_knowledge_dir_returns_empty_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Knowledge dir doesn't exist — should be a no-op but still update state."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    # Note: no `knowledge/` dir created.
    result = decay.run_sweep()
    assert result.archived == 0
    assert result.kept == 0
    # Sweep-state still written so the cooldown ticks.
    assert decay.read_last_sweep_at() is not None


def test_archive_destination_returns_none_for_outside_path(home: Path) -> None:
    k = home / "knowledge"
    outside = home / "outside" / "foo.md"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("---\nid: x\n---\n\nbody\n")
    assert decay._archive_destination(outside, k) is None


def test_run_sweep_continues_after_unreadable_entry(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry that can't be parsed shouldn't abort the sweep — it gets
    classified (no frontmatter fence → ``skipped``, catalog-style file) and
    the loop continues to other entries."""
    k = home / "knowledge"
    _write_entry(k / "global" / "good.md", id_="good", title="Good", created="2026-01-01")
    bad = k / "global" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not even close to a markdown entry\n")  # no frontmatter

    when = datetime(2026, 5, 27, tzinfo=timezone.utc)
    result = decay.run_sweep(knowledge_dir=k, now=when)
    # Good one archived; the fence-less file is catalog-classified, not an
    # error (see test_run_sweep_counts_readme_catalogs_as_skipped_not_errored).
    assert result.archived == 1
    assert result.skipped == 1
    assert result.errored == 0


def test_run_sweep_catches_unexpected_exception_per_entry(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truly-unexpected exception (e.g. permissions) on one entry
    shouldn't crash the sweep — it goes into the errored bucket."""
    k = home / "knowledge"
    _write_entry(k / "global" / "good.md", id_="good", title="Good", created="2026-01-01")

    # Patch the parser to raise mid-loop.
    original = decay._read_and_parse_frontmatter
    calls: list[Path] = []

    def _flaky(path):
        calls.append(path)
        if "good" in str(path):
            raise RuntimeError("simulated read failure")
        return original(path)

    monkeypatch.setattr(decay, "_read_and_parse_frontmatter", _flaky)
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)
    result = decay.run_sweep(knowledge_dir=k, now=when)
    assert result.errored == 1
    assert result.archived == 0


def test_maybe_run_sweep_short_circuits_when_lock_held(home: Path) -> None:
    """If another sweep is in flight, the second caller bails out."""
    k = home / "knowledge"
    with decay._SWEEP_LOCK:
        # Inside the lock, maybe_run_sweep should refuse to run.
        out = decay.maybe_run_sweep(k)
        assert out is None


def test_parse_iso_date_handles_invalid_strings() -> None:
    assert decay._parse_iso_date(None) is None
    assert decay._parse_iso_date("") is None
    assert decay._parse_iso_date("not-a-date") is None
    assert decay._parse_iso_date("2026-13-99") is None  # invalid month/day
    assert decay._parse_iso_date("2026-05-27") == date(2026, 5, 27)


def test_reinforced_count_handles_bad_values() -> None:
    assert decay._reinforced_count({}) == 0
    assert decay._reinforced_count({"reinforced": None}) == 0
    assert decay._reinforced_count({"reinforced": "not a number"}) == 0
    assert decay._reinforced_count({"reinforced": -3}) == 0  # clamped to 0
    assert decay._reinforced_count({"reinforced": 7}) == 7


def test_arousal_pinned_accepts_both_field_names() -> None:
    """``arousal_pin`` and ``arousal_pinned`` both work — the design
    doc switches between them so accept either."""
    assert decay._is_arousal_pinned({"arousal_pin": True}) is True
    assert decay._is_arousal_pinned({"arousal_pinned": True}) is True
    assert decay._is_arousal_pinned({}) is False
    assert decay._is_arousal_pinned({"arousal_pin": False}) is False


def test_write_last_sweep_at_swallows_oserror(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the sweep-state file can't be written, the function returns
    silently — bookkeeping must never crash the sweep itself."""
    monkeypatch.setattr(Path, "mkdir", _raise_oserror)
    # Should not raise.
    decay.write_last_sweep_at(datetime(2026, 5, 27, tzinfo=timezone.utc))


def _raise_oserror(*args, **kwargs):
    raise OSError("simulated")


def test_atomic_write_entry_swallows_oserror(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The atomic-write helper returns False on disk error (rather than
    propagating) so callers can degrade gracefully."""
    p = home / "knowledge" / "global" / "foo.md"
    _write_entry(p, id_="foo", title="Foo")
    monkeypatch.setattr(Path, "write_text", _raise_oserror)
    ok = decay._atomic_write_entry(p, {"id": "foo"}, "body")
    assert ok is False


def test_archive_entry_returns_false_on_destination_write_failure(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the archive destination write fails, the source must stay
    in place (no partial archive)."""
    k = home / "knowledge"
    p = k / "global" / "old.md"
    _write_entry(p, id_="old", title="Old", created="2026-01-01")
    parsed = decay._read_and_parse_frontmatter(p)
    assert parsed is not None
    fm, _raw, body = parsed
    monkeypatch.setattr(Path, "write_text", _raise_oserror)
    ok = decay._archive_entry(p, k, fm, body, today_iso="2026-05-27")
    assert ok is False
    # Source untouched.
    assert p.exists()


def test_iter_entries_skips_top_level_admin_files(home: Path) -> None:
    """index.md and log.md at the top level are catalog files, not
    user entries — the sweep doesn't churn them."""
    k = home / "knowledge"
    (k / "index.md").write_text("---\nid: index\n---\n\n# Index\n")
    (k / "log.md").write_text("---\nid: log\n---\n\n# Log\n")
    _write_entry(k / "global" / "real.md", id_="real", title="Real", created="2026-01-01")
    entries = decay._iter_entries(k)
    names = {e.name for e in entries}
    assert "real.md" in names
    assert "index.md" not in names
    assert "log.md" not in names


def test_read_last_sweep_at_returns_none_for_wrong_field_type(home: Path) -> None:
    """JSON with last_sweep_at as a non-string should be rejected."""
    (home / "sweep-state.json").write_text('{"last_sweep_at": 12345}')
    assert decay.read_last_sweep_at() is None


def test_read_last_sweep_at_returns_none_for_unparseable_iso(home: Path) -> None:
    """A string that isn't ISO format should be rejected."""
    (home / "sweep-state.json").write_text('{"last_sweep_at": "tuesday"}')
    assert decay.read_last_sweep_at() is None


def test_read_last_sweep_at_returns_none_when_top_level_not_dict(home: Path) -> None:
    """JSON array at top level is not a sweep-state record."""
    (home / "sweep-state.json").write_text('["not", "a", "dict"]')
    assert decay.read_last_sweep_at() is None


def test_archive_entry_returns_true_even_when_source_unlink_fails(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the destination write succeeded but source removal failed,
    the archive is effectively done — return True and log a warning."""
    k = home / "knowledge"
    p = k / "global" / "old.md"
    _write_entry(p, id_="old", title="Old", created="2026-01-01")
    parsed = decay._read_and_parse_frontmatter(p)
    assert parsed is not None
    fm, _raw, body = parsed

    real_unlink = Path.unlink

    def _selective_unlink_failure(self, *args, **kwargs):
        # Only fail when unlinking the source entry — let temp-file
        # cleanup work normally.
        if str(self) == str(p):
            raise OSError("simulated unlink failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _selective_unlink_failure)
    ok = decay._archive_entry(p, k, fm, body, today_iso="2026-05-27")
    # Destination is in place, so the archival did happen.
    assert ok is True
    archived = k / "_archive" / "global" / "old.md"
    assert archived.exists()


def test_archive_entry_returns_false_when_destination_resolution_fails(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path that doesn't resolve under the knowledge dir gets a
    None destination and skips archival."""
    k = home / "knowledge"
    # Create entry OUTSIDE the knowledge dir.
    outside = home / "loose" / "stray.md"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("---\nid: stray\ncreated: '2026-01-01'\n---\n\nbody\n")
    parsed = decay._read_and_parse_frontmatter(outside)
    assert parsed is not None
    fm, _raw, body = parsed
    ok = decay._archive_entry(outside, k, fm, body, today_iso="2026-05-27")
    assert ok is False


def test_archive_entry_returns_false_on_mkdir_failure(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k = home / "knowledge"
    p = k / "global" / "old.md"
    _write_entry(p, id_="old", title="Old", created="2026-01-01")
    parsed = decay._read_and_parse_frontmatter(p)
    assert parsed is not None
    fm, _raw, body = parsed
    monkeypatch.setattr(Path, "mkdir", _raise_oserror)
    ok = decay._archive_entry(p, k, fm, body, today_iso="2026-05-27")
    assert ok is False


# ── run_sweep: catalog files vs real parse errors ────────────────────


def test_run_sweep_counts_readme_catalogs_as_skipped_not_errored(home: Path) -> None:
    """Folder READMEs have no frontmatter by design — they are catalog files,
    not decayable entries. They must land in ``skipped``: counting them as
    ``errored`` buries real parse failures under a constant ~25%-of-library
    noise floor (observed: errored=142 with exactly 142 READMEs on disk)."""
    k = home / "knowledge"
    _write_entry(k / "global" / "ok.md", id_="ok", title="Ok", created="2026-05-20")
    readme = k / "global" / "README.md"
    readme.write_text("# Catalog\n\n<!-- ULTAN:children (auto) -->\n", encoding="utf-8")
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)

    result = decay.run_sweep(knowledge_dir=k, now=when)

    assert result.skipped == 1
    assert result.errored == 0
    assert result.kept == 1


def test_run_sweep_counts_malformed_frontmatter_as_errored(home: Path) -> None:
    """A file WITH a frontmatter fence that fails to parse is a real error
    and must stay in ``errored`` (and get a per-file warning)."""
    k = home / "knowledge"
    bad = k / "global" / "bad.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("---\n: not: [valid yaml\n---\nbody\n", encoding="utf-8")
    when = datetime(2026, 5, 27, tzinfo=timezone.utc)

    result = decay.run_sweep(knowledge_dir=k, now=when)

    assert result.errored == 1
    assert result.skipped == 0
