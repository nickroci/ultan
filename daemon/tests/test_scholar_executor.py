"""Tests for the deterministic Scholar executor: each action type applied
to a fixture library, plus index.md / log.md maintenance and the
failure-tolerant batch behaviour."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from agent_mem_daemon import _validation, scholar_executor
from agent_mem_daemon._schemas import ScholarDecisions

from .conftest import scholar_entry_body, seed_scholar_tree

NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)

# A full set of server-owned bookkeeping values, well clear of the template
# defaults, so a clobber back to 0/absent is unambiguous in assertions.
PRESERVED_COUNTERS: Dict[str, Any] = {
    "fired": 8,
    "fired-helpful": 5,
    "last_fired_helpful": "2026-05-10",
    "reinforced": 3,
    "last_reinforced": "2026-05-12",
    "last_surfaced": "2026-05-18",
    "created": "2026-01-02",
}


def _body(id_: str, *, scope: str = "global", extra: str = "") -> str:
    return scholar_entry_body(id_, scope=scope, extra=extra)


def _seed(tmp_path: Path) -> Path:
    return seed_scholar_tree(tmp_path)


def _decisions(*actions) -> ScholarDecisions:
    return ScholarDecisions.model_validate({"actions": list(actions), "interrupts_processed": []})


def _apply(k: Path, *actions) -> scholar_executor.ExecResult:
    return scholar_executor.apply_decisions(_decisions(*actions), k, session_id="s1", now=NOW)


# ── write_entry ───────────────────────────────────────────────────────


def test_write_entry_creates_file_and_index_and_log(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
    )
    assert res.counts["actions_applied"] == 1
    assert res.counts["write_entry"] == 1
    assert (k / "global" / "python" / "new.md").exists()
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert "[[global/python/new]]" in index_md
    assert "| global |" in index_md  # scope column derived from frontmatter
    log_md = (k / "log.md").read_text(encoding="utf-8")
    assert "write_entry | global/python/new.md" in log_md


def test_write_entry_replaces_existing_index_row(tmp_path: Path):
    k = _seed(tmp_path)
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
    )
    # A second write to the same path must replace, not duplicate, the row.
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new", extra=" updated"),
            "reasoning": "r",
        },
    )
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert index_md.count("[[global/python/new]]") == 1


# ── update_entry ──────────────────────────────────────────────────────


def test_update_entry_overwrites_body(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "update_entry",
            "path": "global/python/use-uv.md",
            "new_body": _body("use-uv", extra=" Now with a new clause."),
            "reasoning": "r",
        },
    )
    assert res.counts["update_entry"] == 1
    assert "new clause" in (k / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")


# ── server-owned counter preservation (Defect 1 regression) ───────────


def _entry_with_counters(id_: str, *, scope: str = "global", extra: str = "") -> str:
    """``scholar_entry_body`` with the full server-owned counter set grafted
    into the frontmatter — represents an entry the daemon has already bumped
    on disk before the Scholar's rewrite lands."""
    fm, body = scholar_executor._split_body(scholar_entry_body(id_, scope=scope, extra=extra))
    fm.update(PRESERVED_COUNTERS)
    return scholar_executor._reserialise(fm, body)


def _read_fm(path: Path) -> Dict[str, Any]:
    return _validation.parse_frontmatter(path.read_text(encoding="utf-8"))


def _assert_counters_preserved(path: Path) -> None:
    fm = _read_fm(path)
    for key, want in PRESERVED_COUNTERS.items():
        assert str(fm.get(key)) == str(want), f"{key}: {fm.get(key)!r} != {want!r}"


def test_update_entry_preserves_server_owned_counters(tmp_path: Path):
    """The key regression: an update whose new_body clobbers the counters back
    to the template defaults must NOT lose the on-disk server-owned values —
    while the LLM-owned prose/title DOES take effect."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    target.write_text(_entry_with_counters("use-uv"), encoding="utf-8")
    # The model emits the template defaults (fired/fired-helpful 0, no
    # reinforced) plus genuinely new prose + title.
    clobber = scholar_entry_body("use-uv", extra=" A freshly rewritten clause.")
    clobber = clobber.replace('title: "use-uv"', 'title: "Use uv, rewritten"')
    res = _apply(
        k,
        {
            "action": "update_entry",
            "path": "global/python/use-uv.md",
            "new_body": clobber,
            "reasoning": "r",
        },
    )
    assert res.counts["update_entry"] == 1
    _assert_counters_preserved(target)
    # Prose and title stay LLM-owned. (Frontmatter is re-dumped when counters
    # are grafted back, so assert on the parsed value, not a raw quoted line.)
    text = target.read_text(encoding="utf-8")
    assert "A freshly rewritten clause." in text
    assert _read_fm(target)["title"] == "Use uv, rewritten"


def test_write_entry_brand_new_keeps_defaults(tmp_path: Path):
    """A write_entry to a path with no pre-existing file writes the body as-is
    (defaults stand, no spurious preservation, no crash)."""
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/brand-new.md",
            "body": scholar_entry_body("brand-new"),
            "reasoning": "r",
        },
    )
    assert res.counts["write_entry"] == 1
    fm = _read_fm(k / "global" / "python" / "brand-new.md")
    # Template defaults survive untouched; no counter was invented.
    assert str(fm["fired"]) == "0"
    assert str(fm["fired-helpful"]) == "0"
    assert "reinforced" not in fm
    assert "last_surfaced" not in fm


def test_write_entry_replacing_existing_preserves_counters(tmp_path: Path):
    """A write_entry that overwrites an existing entry is a replace too — its
    server-owned counters must be preserved like an update."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    target.write_text(_entry_with_counters("use-uv"), encoding="utf-8")
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/use-uv.md",
            "body": scholar_entry_body("use-uv", extra=" rewritten via write"),
            "reasoning": "r",
        },
    )
    _assert_counters_preserved(target)
    assert "rewritten via write" in target.read_text(encoding="utf-8")


def test_merge_preserves_target_counters_when_overwriting(tmp_path: Path):
    """Merge that overwrites an existing target in place keeps the target's
    server-owned counters (the merged body's clobbered values are discarded)."""
    k = _seed(tmp_path)
    # target_path already exists with elevated counters.
    target = k / "global" / "python" / "merged.md"
    target.write_text(_entry_with_counters("merged"), encoding="utf-8")
    (k / "global" / "python" / "a.md").write_text(scholar_entry_body("a"), encoding="utf-8")
    res = _apply(
        k,
        {
            "action": "merge_entries",
            "source_paths": ["global/python/a.md"],
            "target_path": "global/python/merged.md",
            "target_body": scholar_entry_body("merged", extra=" merged prose"),
            "reasoning": "r",
        },
    )
    assert res.counts["merge_entries"] == 1
    _assert_counters_preserved(target)
    assert "merged prose" in target.read_text(encoding="utf-8")


def test_deprecate_preserves_counters(tmp_path: Path):
    """Deprecate rewrites frontmatter in place; the server-owned counters of
    the deprecated entry survive (only status/superseded_by/banner change)."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    target.write_text(_entry_with_counters("use-uv"), encoding="utf-8")
    (k / "global" / "python" / "newer.md").write_text(scholar_entry_body("newer"), encoding="utf-8")
    res = _apply(
        k,
        {
            "action": "deprecate_entry",
            "path": "global/python/use-uv.md",
            "superseded_by": "global/python/newer.md",
            "reasoning": "r",
        },
    )
    assert res.counts["deprecate_entry"] == 1
    _assert_counters_preserved(target)
    text = target.read_text(encoding="utf-8")
    assert "status: deprecated" in text


def test_abstract_preserves_parent_counters_when_overwriting(tmp_path: Path):
    """If abstraction overwrites an existing parent entry, that parent's
    server-owned counters are preserved."""
    k = _seed_two_children(tmp_path)
    parent = k / "global" / "conventions" / "likes-tooling.md"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text(_entry_with_counters("likes-tooling"), encoding="utf-8")
    res = _apply(
        k,
        {
            "action": "abstract_entries",
            "child_paths": ["global/python/use-uv.md", "global/python/lint.md"],
            "parent_path": "global/conventions/likes-tooling.md",
            "parent_title": "Likes tooling",
            "parent_body": _abstraction_body("likes-tooling"),
            "reasoning": "r",
        },
    )
    assert res.counts["abstract_entries"] == 1
    _assert_counters_preserved(parent)


# ── merge_entries ─────────────────────────────────────────────────────


def test_merge_entries_writes_target_and_archives_sources(tmp_path: Path):
    k = _seed(tmp_path)
    (k / "global" / "python" / "a.md").write_text(_body("a"), encoding="utf-8")
    (k / "global" / "python" / "b.md").write_text(_body("b"), encoding="utf-8")
    res = _apply(
        k,
        {
            "action": "merge_entries",
            "source_paths": ["global/python/a.md", "global/python/b.md"],
            "target_path": "global/python/merged.md",
            "target_body": _body("merged"),
            "reasoning": "r",
        },
    )
    assert res.counts["merge_entries"] == 1
    assert (k / "global" / "python" / "merged.md").exists()
    # Sources archived (moved out), not left in place.
    assert not (k / "global" / "python" / "a.md").exists()
    assert (k / "_archive" / "global" / "python" / "a.md").exists()
    archived = (k / "_archive" / "global" / "python" / "a.md").read_text(encoding="utf-8")
    assert "status: stale" in archived


# ── move_entry ────────────────────────────────────────────────────────


def test_move_entry_relocates_and_rewrites_index(tmp_path: Path):
    k = _seed(tmp_path)
    # Seed an index row for the entry being moved.
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/use-uv.md",
            "body": _body("use-uv"),
            "reasoning": "seed index",
        },
    )
    res = _apply(
        k,
        {
            "action": "move_entry",
            "from_path": "global/python/use-uv.md",
            "to_path": "global/tooling/use-uv.md",
        },
    )
    assert res.counts["move_entry"] == 1
    assert not (k / "global" / "python" / "use-uv.md").exists()
    assert (k / "global" / "tooling" / "use-uv.md").exists()
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert "[[global/tooling/use-uv]]" in index_md
    assert "[[global/python/use-uv]]" not in index_md


def test_move_entry_failure_recorded(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "move_entry",
            "from_path": "global/python/does-not-exist.md",
            "to_path": "global/tooling/x.md",
        },
    )
    assert res.counts.get("actions_failed") == 1


# ── archive_entry ─────────────────────────────────────────────────────


def test_archive_entry_moves_and_stamps(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k, {"action": "archive_entry", "path": "global/python/use-uv.md", "reasoning": "r"}
    )
    assert res.counts["archive_entry"] == 1
    assert not (k / "global" / "python" / "use-uv.md").exists()
    archived = (k / "_archive" / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")
    assert "status: stale" in archived
    assert "2026-05-19" in archived
    assert "archived:" in archived


def test_archive_missing_entry_recorded_as_failure(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(k, {"action": "archive_entry", "path": "global/python/ghost.md", "reasoning": "r"})
    assert res.counts.get("actions_failed") == 1


# ── deprecate_entry ───────────────────────────────────────────────────


def test_deprecate_entry_marks_in_place(tmp_path: Path):
    k = _seed(tmp_path)
    (k / "global" / "python" / "newer.md").write_text(_body("newer"), encoding="utf-8")
    res = _apply(
        k,
        {
            "action": "deprecate_entry",
            "path": "global/python/use-uv.md",
            "superseded_by": "global/python/newer.md",
            "reasoning": "r",
        },
    )
    assert res.counts["deprecate_entry"] == 1
    # File kept in place so inbound wikilinks still resolve.
    text = (k / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")
    assert "status: deprecated" in text
    assert "superseded_by: global/python/newer.md" in text
    assert "Superseded by [[global/python/newer]]" in text


def test_deprecate_missing_entry_recorded_as_failure(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        {
            "action": "deprecate_entry",
            "path": "global/python/ghost.md",
            "superseded_by": "global/python/use-uv.md",
            "reasoning": "r",
        },
    )
    assert res.counts.get("actions_failed") == 1


# ── abstract_entries ──────────────────────────────────────────────────


def _abstraction_body(id_: str, *, scope: str = "global") -> str:
    """A valid parent-abstraction body — ``type: abstraction`` + child links."""
    return _body(id_, scope=scope).replace("type: lesson", "type: abstraction") + (
        "\n## Related\n\n- [[global/python/use-uv]]\n- [[global/python/lint]]\n"
    )


def _seed_two_children(tmp_path: Path) -> Path:
    k = _seed(tmp_path)
    (k / "global" / "python" / "lint.md").write_text(_body("lint"), encoding="utf-8")
    return k


def test_abstract_entries_writes_parent_backlinks_children_and_index(tmp_path: Path):
    k = _seed_two_children(tmp_path)
    res = _apply(
        k,
        {
            "action": "abstract_entries",
            "child_paths": ["global/python/use-uv.md", "global/python/lint.md"],
            "parent_path": "global/conventions/likes-tooling.md",
            "parent_title": "Likes tooling",
            "parent_body": _abstraction_body("likes-tooling"),
            "reasoning": "r",
        },
    )
    assert res.counts["actions_applied"] == 1
    assert res.counts["abstract_entries"] == 1
    # Parent written with type: abstraction.
    parent = (k / "global" / "conventions" / "likes-tooling.md").read_text(encoding="utf-8")
    assert "type: abstraction" in parent
    # Index row for the parent.
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert "[[global/conventions/likes-tooling]]" in index_md
    # Reverse backlink added into each child; children otherwise intact.
    for child in ("use-uv", "lint"):
        text = (k / "global" / "python" / f"{child}.md").read_text(encoding="utf-8")
        assert "[[global/conventions/likes-tooling]]" in text
        assert "A real body sentence." in text  # original body preserved
    # Children NOT archived or moved — still in place.
    assert (k / "global" / "python" / "use-uv.md").exists()
    assert (k / "global" / "python" / "lint.md").exists()
    assert not (k / "_archive" / "global" / "python" / "use-uv.md").exists()
    # Log appended.
    log_md = (k / "log.md").read_text(encoding="utf-8")
    assert "abstract_entries | global/conventions/likes-tooling.md" in log_md


def test_abstract_entries_backlink_idempotent(tmp_path: Path):
    """Re-running the same abstraction does not pile up duplicate backlinks."""
    k = _seed_two_children(tmp_path)
    action = {
        "action": "abstract_entries",
        "child_paths": ["global/python/use-uv.md", "global/python/lint.md"],
        "parent_path": "global/conventions/likes-tooling.md",
        "parent_title": "Likes tooling",
        "parent_body": _abstraction_body("likes-tooling"),
        "reasoning": "r",
    }
    _apply(k, action)
    _apply(k, action)
    text = (k / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")
    assert text.count("[[global/conventions/likes-tooling]]") == 1


def test_abstract_entries_appends_related_when_absent(tmp_path: Path):
    """A child without a Related section gets a fresh one appended."""
    k = _seed_two_children(tmp_path)
    # use-uv.md (from seed) has no Related section.
    _apply(
        k,
        {
            "action": "abstract_entries",
            "child_paths": ["global/python/use-uv.md", "global/python/lint.md"],
            "parent_path": "global/conventions/likes-tooling.md",
            "parent_title": "Likes tooling",
            "parent_body": _abstraction_body("likes-tooling"),
            "reasoning": "r",
        },
    )
    text = (k / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")
    assert "## Related" in text
    assert "[[global/conventions/likes-tooling]]" in text


def test_abstract_entries_appends_to_existing_related(tmp_path: Path):
    """A child that already has a Related section gets the new bullet under it
    (the existing link is preserved)."""
    k = _seed_two_children(tmp_path)
    child = k / "global" / "python" / "use-uv.md"
    child.write_text(
        _body("use-uv") + "\n## Related\n\n- [[global/python/lint]] — sibling\n",
        encoding="utf-8",
    )
    _apply(
        k,
        {
            "action": "abstract_entries",
            "child_paths": ["global/python/use-uv.md", "global/python/lint.md"],
            "parent_path": "global/conventions/likes-tooling.md",
            "parent_title": "Likes tooling",
            "parent_body": _abstraction_body("likes-tooling"),
            "reasoning": "r",
        },
    )
    text = child.read_text(encoding="utf-8")
    assert text.count("## Related") == 1  # not a second section
    assert "[[global/python/lint]] — sibling" in text  # existing bullet kept
    assert "[[global/conventions/likes-tooling]]" in text  # new backlink added


def test_abstract_entries_unreadable_child_recorded_not_fatal(tmp_path: Path):
    """A missing child is noted but the parent + readable backlinks still apply."""
    k = _seed_two_children(tmp_path)
    res = _apply(
        k,
        {
            "action": "abstract_entries",
            "child_paths": ["global/python/use-uv.md", "global/python/ghost.md"],
            "parent_path": "global/conventions/likes-tooling.md",
            "parent_title": "Likes tooling",
            "parent_body": _abstraction_body("likes-tooling"),
            "reasoning": "r",
        },
    )
    # The action still succeeds (parent written) — integrity-first.
    assert res.counts["abstract_entries"] == 1
    assert (k / "global" / "conventions" / "likes-tooling.md").exists()
    assert any("ghost.md" in n and "unreadable" in n for n in res.notes)
    # Readable child still backlinked.
    text = (k / "global" / "python" / "use-uv.md").read_text(encoding="utf-8")
    assert "[[global/conventions/likes-tooling]]" in text


# ── _add_parent_backlink (helper edge cases) ──────────────────────────


def test_add_parent_backlink_inserts_before_following_heading():
    """Bullet lands at the end of the Related section, before a later
    heading — and a trailing blank line inside the section is trimmed."""
    body = (
        "# Title\n\nProse.\n\n## Related\n\n- [[global/python/lint]]\n\n## Notes\n\nMore prose.\n"
    )
    out = scholar_executor._add_parent_backlink(body, "global/conventions/parent")
    related = out.index("## Related")
    notes = out.index("## Notes")
    link = out.index("[[global/conventions/parent]]")
    # New bullet sits inside the Related section (after the heading, before Notes).
    assert related < link < notes
    assert out.count("## Related") == 1


def test_add_parent_backlink_body_without_trailing_newline():
    """A body that does not end in a newline still gets a Related section."""
    body = "# Title\n\nProse with no trailing newline."
    out = scholar_executor._add_parent_backlink(body, "global/conventions/parent")
    assert "## Related" in out
    assert "[[global/conventions/parent]]" in out


def test_add_parent_backlink_idempotent_helper():
    body = "# Title\n\n## Related\n\n- [[global/conventions/parent]]\n"
    assert scholar_executor._add_parent_backlink(body, "global/conventions/parent") == body


# ── batch behaviour ───────────────────────────────────────────────────


def test_one_bad_action_does_not_abort_batch(tmp_path: Path):
    k = _seed(tmp_path)
    res = _apply(
        k,
        # good
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
        # bad (archive of a non-existent file)
        {"action": "archive_entry", "path": "global/python/ghost.md", "reasoning": "r"},
    )
    assert res.counts.get("write_entry") == 1
    assert res.counts.get("actions_failed") == 1
    assert (k / "global" / "python" / "new.md").exists()


def test_empty_decisions_no_op(tmp_path: Path):
    k = _seed(tmp_path)
    res = scholar_executor.apply_decisions(_decisions(), k, session_id="s1", now=NOW)
    assert res.counts == {}
    assert res.notes == []


def test_index_created_when_absent(tmp_path: Path):
    k = tmp_path / "knowledge"
    k.mkdir()
    _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/x.md",
            "body": _body("x"),
            "reasoning": "r",
        },
    )
    index_md = (k / "index.md").read_text(encoding="utf-8")
    assert "Knowledge Index" in index_md
    assert "[[global/x]]" in index_md


def test_dispatch_handler_exception_recorded(tmp_path: Path, monkeypatch):
    """If a handler raises unexpectedly, apply_decisions records a failure
    rather than propagating."""
    k = _seed(tmp_path)

    def _boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(scholar_executor, "_do_write", _boom)
    res = _apply(
        k,
        {
            "action": "write_entry",
            "path": "global/python/new.md",
            "body": _body("new"),
            "reasoning": "r",
        },
    )
    assert res.counts.get("actions_failed") == 1


# ── reconsolidation (drift) counter bookkeeping ───────────────────────


def _drift_update(extra: str = " A reconsolidated clause.") -> Dict[str, Any]:
    """A drift update_entry on the seeded ``use-uv`` entry."""
    return {
        "action": "update_entry",
        "path": "global/python/use-uv.md",
        "new_body": _body("use-uv", extra=extra),
        "salience_signal": "drift",
        "existing_entry": "global/python/use-uv.md",
        "reasoning": "r",
    }


def test_drift_update_increments_reconsolidated_from_zero(tmp_path: Path):
    """A drift update on a never-reconsolidated entry sets reconsolidated=1 and
    stamps last_reconsolidated with the run date."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    _apply(k, _drift_update())
    fm = _read_fm(target)
    assert str(fm["reconsolidated"]) == "1"
    assert str(fm["last_reconsolidated"]) == NOW.date().isoformat()
    assert "reconsolidated clause" in target.read_text(encoding="utf-8")


def test_second_drift_update_increments_again(tmp_path: Path):
    """Reconsolidation count is authoritative from the prior on-disk value: two
    drift updates leave reconsolidated=2 regardless of what the model emits."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    _apply(k, _drift_update(extra=" first"))
    _apply(k, _drift_update(extra=" second"))
    assert str(_read_fm(target)["reconsolidated"]) == "2"


def test_drift_update_increments_existing_count(tmp_path: Path):
    """Drift update on an entry that already carries reconsolidated=3 bumps it
    to 4 (prior-on-disk + 1), never trusting the emitted body's value."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    fm, body = scholar_executor._split_body(scholar_entry_body("use-uv"))
    fm["reconsolidated"] = 3
    target.write_text(scholar_executor._reserialise(fm, body), encoding="utf-8")
    _apply(k, _drift_update())
    assert str(_read_fm(target)["reconsolidated"]) == "4"


def test_drift_update_preserves_other_counters(tmp_path: Path):
    """A drift reconsolidation still preserves the other server-owned counters
    (fired/reinforced/…) — it only authoritatively rewrites the reconsolidation
    pair."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    target.write_text(_entry_with_counters("use-uv"), encoding="utf-8")
    _apply(k, _drift_update())
    _assert_counters_preserved(target)
    assert str(_read_fm(target)["reconsolidated"]) == "1"


def test_non_drift_update_preserves_reconsolidated_without_bumping(tmp_path: Path):
    """An ordinary (non-drift) update must NOT increment reconsolidated — it
    only preserves the existing value so a repair/edit can't reset the history
    or spuriously inflate it."""
    k = _seed(tmp_path)
    target = k / "global" / "python" / "use-uv.md"
    fm, body = scholar_executor._split_body(scholar_entry_body("use-uv"))
    fm["reconsolidated"] = 3
    fm["last_reconsolidated"] = "2026-05-01"
    target.write_text(scholar_executor._reserialise(fm, body), encoding="utf-8")
    _apply(
        k,
        {
            "action": "update_entry",
            "path": "global/python/use-uv.md",
            "new_body": _body("use-uv", extra=" a plain non-drift edit"),
            "reasoning": "r",
        },
    )
    out = _read_fm(target)
    assert str(out["reconsolidated"]) == "3"  # preserved, not bumped
    assert str(out["last_reconsolidated"]) == "2026-05-01"
