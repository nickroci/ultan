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


# ── ensure_alias_for_cwd — auto-bootstrap behaviour ────────────────────


def _make_git_repo(path: Path) -> Path:
    """Mark a directory as a git repo for the walk-up detection. We
    don't need an actual git history — just the ``.git`` entry."""
    (path / ".git").mkdir(parents=True, exist_ok=True)
    return path


def _make_bucket(home: Path, name: str) -> Path:
    """Materialise an empty library bucket."""
    bucket = home / "knowledge" / "projects" / name
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket


def test_ensure_alias_writes_mapping_when_bucket_exists(tmp_path: Path):
    """Repo at ``cwd``, git remote-derived slug doesn't match any
    bucket directly, but a bucket with the repo-root basename exists —
    auto-add the mapping."""
    home = tmp_path / "agent-mem-home"
    home.mkdir()
    repo = _make_git_repo(tmp_path / "agent-mem")
    _make_bucket(home, "agent-mem")

    result = aliases.ensure_alias_for_cwd(home, repo, "github.com-nickroci-ultan")

    assert result == ("agent-mem", "github.com-nickroci-ultan")
    on_disk = aliases.load_aliases(home)
    assert on_disk == {"agent-mem": "github.com-nickroci-ultan"}


def test_ensure_alias_finds_repo_root_from_subfolder(tmp_path: Path):
    """Starting in ``agent-mem/daemon`` (subfolder) should still
    resolve to bucket ``agent-mem`` via the walk-up to ``.git``."""
    home = tmp_path / "agent-mem-home"
    home.mkdir()
    repo = _make_git_repo(tmp_path / "agent-mem")
    sub = repo / "daemon"
    sub.mkdir()
    _make_bucket(home, "agent-mem")

    result = aliases.ensure_alias_for_cwd(home, sub, "github.com-nickroci-ultan")
    assert result == ("agent-mem", "github.com-nickroci-ultan")


def test_ensure_alias_noop_when_no_git(tmp_path: Path):
    """No git -> slug already came from cwd basename -> if a matching
    bucket exists it works without an alias; if not, we have nothing
    to add. Either way the file stays empty."""
    home = tmp_path / "agent-mem-home"
    home.mkdir()
    repo = tmp_path / "agent-mem"  # NB: no .git
    repo.mkdir()
    _make_bucket(home, "agent-mem")

    result = aliases.ensure_alias_for_cwd(home, repo, "agent-mem")
    assert result is None
    assert not aliases.aliases_path(home).exists()


def test_ensure_alias_noop_when_slug_already_matches_some_bucket(tmp_path: Path):
    """If the slug is already a bucket name (direct match) we don't
    need an alias entry — bail out."""
    home = tmp_path / "agent-mem-home"
    home.mkdir()
    repo = _make_git_repo(tmp_path / "some-other-folder")
    _make_bucket(home, "agent-mem")
    # Slug equals an existing bucket — direct match, no alias needed.
    result = aliases.ensure_alias_for_cwd(home, repo, "agent-mem")
    assert result is None


def test_ensure_alias_noop_when_alias_already_set(tmp_path: Path):
    """Existing alias entry for the bucket is sacred — never overwrite."""
    home = tmp_path / "agent-mem-home"
    home.mkdir()
    repo = _make_git_repo(tmp_path / "agent-mem")
    _make_bucket(home, "agent-mem")
    # Pre-populate with a different slug.
    aliases_file = aliases.aliases_path(home)
    aliases_file.parent.mkdir(parents=True, exist_ok=True)
    aliases_file.write_text('{"agent-mem": "github.com-someone-else-ultan"}', encoding="utf-8")

    result = aliases.ensure_alias_for_cwd(home, repo, "github.com-nickroci-ultan")
    assert result is None
    # File untouched.
    assert aliases.load_aliases(home) == {"agent-mem": "github.com-someone-else-ultan"}


def test_ensure_alias_noop_when_no_matching_bucket(tmp_path: Path):
    """Repo folder name doesn't correspond to any bucket — bail out
    silently rather than create a bucket-less alias."""
    home = tmp_path / "agent-mem-home"
    home.mkdir()
    repo = _make_git_repo(tmp_path / "brand-new-project")
    # Only an unrelated bucket exists.
    _make_bucket(home, "vol-predictor")

    result = aliases.ensure_alias_for_cwd(home, repo, "github.com-x-brand-new-project")
    assert result is None
    assert not aliases.aliases_path(home).exists()


def test_ensure_alias_adds_when_git_arrives_later(tmp_path: Path):
    """Simulate the 'I added a git remote after the bucket existed'
    flow: first session was no-git so no alias file; then user runs
    `git init` and sets a remote; next session should auto-add."""
    home = tmp_path / "agent-mem-home"
    home.mkdir()
    repo = tmp_path / "agent-mem"
    repo.mkdir()
    _make_bucket(home, "agent-mem")

    # First session: no git, slug == "agent-mem" -> no alias added.
    assert aliases.ensure_alias_for_cwd(home, repo, "agent-mem") is None

    # User runs `git init`; slug now derives from the remote.
    (repo / ".git").mkdir()
    result = aliases.ensure_alias_for_cwd(home, repo, "github.com-nickroci-ultan")
    assert result == ("agent-mem", "github.com-nickroci-ultan")
