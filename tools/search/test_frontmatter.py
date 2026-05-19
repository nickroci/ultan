"""Tests for the frontmatter read/write utility."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from frontmatter import (
    FrontmatterError,
    bump_updated,
    read,
    set_status,
    write,
)


def _sample_doc(status: str = "provisional") -> str:
    return (
        "---\n"
        "id: factory-pattern-for-apis\n"
        "type: lesson\n"
        "scope: global\n"
        f"status: {status}\n"
        "confidence: 0.72\n"
        "applies-when: |\n"
        "  designing or building any new API\n"
        "  decisions about how clients construct service objects\n"
        "keywords: [factory, paradigm, api]\n"
        "created: 2026-05-19\n"
        "updated: 2026-05-19\n"
        "fired: 0\n"
        "fired-helpful: 0\n"
        "---\n"
        "\n"
        "# Use factory pattern for new APIs\n"
        "\n"
        "**Rule:** Don't `new` your services.\n"
    )


def test_read_round_trip_preserves_body(tmp_path: Path) -> None:
    p = tmp_path / "entry.md"
    p.write_text(_sample_doc(), encoding="utf-8")
    fm, body = read(p)
    assert fm["id"] == "factory-pattern-for-apis"
    assert fm["status"] == "provisional"
    assert fm["keywords"] == ["factory", "paradigm", "api"]
    # The regex consumes the closing `---\n`, so body starts immediately after.
    assert "# Use factory pattern for new APIs" in body
    assert body.lstrip().startswith("# Use factory pattern for new APIs")


def test_write_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "entry.md"
    p.write_text(_sample_doc(), encoding="utf-8")
    fm, body = read(p)
    fm["confidence"] = 0.91
    write(p, fm, body)
    fm2, body2 = read(p)
    assert fm2["confidence"] == 0.91
    # All required fields survive.
    for key in ("id", "type", "scope", "status", "applies-when", "keywords"):
        assert key in fm2
    assert body2 == body


def test_write_preserves_block_scalar_for_multiline_strings(tmp_path: Path) -> None:
    p = tmp_path / "entry.md"
    p.write_text(_sample_doc(), encoding="utf-8")
    fm, body = read(p)
    # applies-when comes in as a single multi-line string.
    assert isinstance(fm["applies-when"], str)
    assert "\n" in fm["applies-when"]
    write(p, fm, body)
    raw = p.read_text(encoding="utf-8")
    # The block-scalar `|` form should be emitted, not a giant quoted string.
    assert "applies-when: |" in raw


def test_set_status_changes_status_and_bumps_updated(tmp_path: Path) -> None:
    p = tmp_path / "entry.md"
    p.write_text(_sample_doc(status="provisional"), encoding="utf-8")
    set_status(p, "confirmed", today="2027-01-01")
    fm, _ = read(p)
    assert fm["status"] == "confirmed"
    assert fm["updated"] == "2027-01-01"


def test_set_status_rejects_garbage(tmp_path: Path) -> None:
    p = tmp_path / "entry.md"
    p.write_text(_sample_doc(), encoding="utf-8")
    with pytest.raises(ValueError):
        set_status(p, "yolo")


def test_bump_updated_uses_today_by_default(tmp_path: Path) -> None:
    p = tmp_path / "entry.md"
    p.write_text(_sample_doc(), encoding="utf-8")
    bump_updated(p)
    fm, _ = read(p)
    assert fm["updated"] == date.today().isoformat()


def test_write_is_atomic_no_partial_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    """If the rename step fails we shouldn't leave a half-written file behind.

    Simulate failure by patching ``os.replace`` to raise. The original file
    must remain intact; the tmp file must be cleaned up.
    """
    p = tmp_path / "entry.md"
    original = _sample_doc()
    p.write_text(original, encoding="utf-8")

    real_replace = os.replace

    def boom(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", boom)

    fm, body = read(p)
    fm["confidence"] = 0.99
    with pytest.raises(OSError):
        write(p, fm, body)

    # Original file untouched.
    assert p.read_text(encoding="utf-8") == original
    # Tmp files cleaned up.
    leftover = [child for child in tmp_path.iterdir() if child.name.endswith(".tmp")]
    assert not leftover, f"left-over tmp files: {leftover}"

    # Restore so the test process can continue normally.
    monkeypatch.setattr(os, "replace", real_replace)


def test_read_missing_frontmatter_returns_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "no-fm.md"
    p.write_text("# Just a heading\n\nNo frontmatter here.\n", encoding="utf-8")
    fm, body = read(p)
    assert fm == {}
    assert "Just a heading" in body


def test_read_malformed_frontmatter_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.md"
    p.write_text("---\nid: foo\n  bad: : : indent\n---\nbody\n", encoding="utf-8")
    with pytest.raises(FrontmatterError):
        read(p)


def test_field_ordering_keeps_canonical_keys_first(tmp_path: Path) -> None:
    p = tmp_path / "entry.md"
    p.write_text(_sample_doc(), encoding="utf-8")
    fm, body = read(p)
    # Insert a non-canonical field somewhere in the middle.
    fm["zzz-custom"] = "trailing"
    write(p, fm, body)
    raw = p.read_text(encoding="utf-8")
    # `id` must appear before `status`, which must appear before `zzz-custom`.
    id_pos = raw.find("id:")
    status_pos = raw.find("status:")
    custom_pos = raw.find("zzz-custom:")
    assert 0 <= id_pos < status_pos < custom_pos
