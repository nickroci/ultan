"""Tests for the shared project-alias module.

These pin the contract relied on by both the daemon (priming scope
bonus) and the user-prompt-submit hook (cross-project nudge filter).
"""

from __future__ import annotations

from pathlib import Path

import aliases


def test_load_aliases_missing_file_returns_empty(tmp_path: Path):
    """No file -> empty dict, no exception."""
    assert aliases.load_aliases(tmp_path) == {}


def test_load_aliases_reads_valid_json(tmp_path: Path):
    (tmp_path / "project-aliases.json").write_text(
        '{"agent-mem": "github.com-nickroci-ultan", "fork": "github.com-nickroci-ultan"}',
        encoding="utf-8",
    )
    out = aliases.load_aliases(tmp_path)
    assert out == {
        "agent-mem": "github.com-nickroci-ultan",
        "fork": "github.com-nickroci-ultan",
    }


def test_load_aliases_malformed_json_returns_empty(tmp_path: Path):
    """Bad JSON must not crash the caller — silently fall back to {}."""
    (tmp_path / "project-aliases.json").write_text("not json {", encoding="utf-8")
    assert aliases.load_aliases(tmp_path) == {}


def test_load_aliases_non_object_returns_empty(tmp_path: Path):
    """A JSON list or scalar isn't a valid map — treat as {}."""
    (tmp_path / "project-aliases.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert aliases.load_aliases(tmp_path) == {}


def test_bucket_canonical_slug_defaults_to_bucket_name():
    """Bucket not in the map -> bucket is its own slug."""
    assert aliases.bucket_canonical_slug("agent-mem", {}) == "agent-mem"
    assert aliases.bucket_canonical_slug("vol-predictor", {"other": "x"}) == "vol-predictor"


def test_bucket_canonical_slug_uses_explicit_alias():
    assert (
        aliases.bucket_canonical_slug("agent-mem", {"agent-mem": "github.com-nickroci-ultan"})
        == "github.com-nickroci-ultan"
    )


def test_bucket_canonical_slug_none_in_none_out():
    """Defensive contract: ``None`` propagates so callers don't need
    to pre-check."""
    assert aliases.bucket_canonical_slug(None, {}) is None
    assert aliases.bucket_canonical_slug("", {}) is None
