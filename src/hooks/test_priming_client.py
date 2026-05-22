"""Extended unit tests for the priming client's lexical-fallback path.

The end-to-end socket-roundtrip cases live in
``test_user_prompt_submit.py``; here we exercise the local-fallback
machinery directly so the YAML-lite parser, summary derivation, char-
budget trimming, and bad-file handling are all measured."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import _priming_client  # noqa: E402


def _write_entry(
    path: Path,
    *,
    keywords: list[str],
    applies_when=None,
    title: str | None = None,
    reinforced: int | None = None,
    body: str = "body content",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    lines.append("keywords: [" + ", ".join(keywords) + "]")
    if applies_when is not None:
        if isinstance(applies_when, list):
            lines.append("applies-when: [" + ", ".join(applies_when) + "]")
        else:
            lines.append("applies-when: |")
            for line in applies_when.splitlines() or [applies_when]:
                lines.append(f"  {line}")
    if title is not None:
        lines.append(f'title: "{title}"')
    if reinforced is not None:
        lines.append(f"reinforced: {reinforced}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_priming_empty_knowledge_dir_returns_empty(tmp_path: Path):
    """No knowledge dir on disk → fallback returns ''."""
    out = _priming_client._local_priming(
        "python uv", k=5, char_budget=1500, knowledge_dir=tmp_path / "missing"
    )
    assert out == ""


def test_priming_no_matches_returns_empty(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(kdir / "unrelated.md", keywords=["banana"])
    out = _priming_client._local_priming(
        "completely different topic", k=5, char_budget=1500, knowledge_dir=kdir
    )
    assert out == ""


def test_priming_summary_falls_back_to_title(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    # No applies-when, but title present.
    _write_entry(
        kdir / "t.md",
        keywords=["python"],
        title="Use uv",
        body="python python python uv",
    )
    out = _priming_client._local_priming("python uv", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "Use uv" in out


def test_priming_summary_falls_back_to_stem(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    # No applies-when, no title — summary becomes file stem.
    _write_entry(
        kdir / "named-stem.md",
        keywords=["python"],
        body="python python uv",
    )
    out = _priming_client._local_priming("python uv", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "named stem" in out or "named-stem" in out


def test_priming_list_applies_when_used_for_summary(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(
        kdir / "t.md",
        keywords=["python"],
        applies_when=["installing python deps"],
    )
    out = _priming_client._local_priming("python deps", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "installing python deps" in out


def test_priming_reinforced_marker_appears(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(
        kdir / "t.md",
        keywords=["python"],
        title="Use uv",
        reinforced=7,
    )
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "(×7)" in out


def test_priming_negative_reinforced_clamps_to_zero(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(
        kdir / "t.md",
        keywords=["python"],
        title="x",
        reinforced=-5,
    )
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    # Negative count → no (×N) suffix.
    assert "(×" not in out


def test_priming_invalid_reinforced_treated_as_zero(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    # Manually write so reinforced is non-numeric.
    (kdir / "bad-reinforced.md").write_text(
        "---\nkeywords: [python]\ntitle: x\nreinforced: maybe\n---\npython body\n",
        encoding="utf-8",
    )
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "(×" not in out


def test_priming_skips_unreadable_file(tmp_path: Path, monkeypatch):
    """A file that raises on read shouldn't kill the whole scan."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(kdir / "good.md", keywords=["python"], title="good")
    bad = kdir / "bad.md"
    bad.write_text("---\nkeywords: [python]\ntitle: bad\n---\nbody\n", encoding="utf-8")
    orig = Path.read_text

    def maybe_raise(self, *args, **kwargs):
        if self.name == "bad.md":
            raise OSError("permission denied")
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", maybe_raise)
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "good" in out
    assert "bad" not in out


def test_priming_char_budget_trims_bullets(tmp_path: Path):
    """If the rendered output exceeds char_budget, trailing bullets are
    dropped until it fits."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    # 10 matching entries with long titles.
    for i in range(10):
        _write_entry(
            kdir / f"e{i}.md",
            keywords=["python"],
            title=f"A long title for entry number {i:02d} to push the budget",
        )
    out = _priming_client._local_priming("python", k=10, char_budget=400, knowledge_dir=kdir)
    # Output should respect the budget (give or take a couple of bytes
    # of last-bullet slack).
    assert len(out) <= 500
    assert "## Ultan" in out


def test_priming_char_budget_too_small_keeps_one_bullet(tmp_path: Path):
    """If even one bullet exceeds the budget, we still emit one — the
    fallback is "less is worse than slightly over-budget"."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(kdir / "e.md", keywords=["python"], title="x" * 200)
    out = _priming_client._local_priming("python", k=5, char_budget=50, knowledge_dir=kdir)
    assert "## Ultan" in out


def test_priming_skips_archive_subtrees(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(kdir / "_archive" / "old.md", keywords=["python"], title="archived")
    _write_entry(kdir / "fresh.md", keywords=["python"], title="fresh")
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "fresh" in out
    assert "archived" not in out


def test_priming_skips_top_level_index_log_readme(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "index.md").write_text(
        "---\nkeywords: [python]\ntitle: ix\n---\npython body\n", encoding="utf-8"
    )
    (kdir / "log.md").write_text(
        "---\nkeywords: [python]\ntitle: lg\n---\npython body\n", encoding="utf-8"
    )
    (kdir / "README.md").write_text(
        "---\nkeywords: [python]\ntitle: rd\n---\npython body\n", encoding="utf-8"
    )
    _write_entry(kdir / "real.md", keywords=["python"], title="real-entry")
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "real-entry" in out
    assert "[[index]]" not in out
    assert "[[log]]" not in out


def test_priming_yaml_lite_handles_block_scalar_gt(tmp_path: Path):
    """``key: >`` is an alternate block-scalar marker — treat like ``|``."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "e.md").write_text(
        "---\n"
        "keywords: [python]\n"
        "applies-when: >\n"
        "  installing python\n"
        "title: t\n"
        "---\n"
        "python body\n",
        encoding="utf-8",
    )
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    assert "installing python" in out


def test_priming_no_query_tokens_returns_empty(tmp_path: Path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    _write_entry(kdir / "e.md", keywords=["python"], title="x")
    # Only whitespace/punct => no tokens after tokenisation.
    out = _priming_client._local_priming("!!!", k=5, char_budget=1500, knowledge_dir=kdir)
    assert out == ""


def test_priming_empty_files_list_returns_empty(tmp_path: Path):
    """Knowledge dir exists but contains nothing scannable."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    # Only catalog files (skipped).
    (kdir / "index.md").write_text("# Index", encoding="utf-8")
    out = _priming_client._local_priming("python", k=5, char_budget=1500, knowledge_dir=kdir)
    assert out == ""


def test_wikilink_handles_non_relative_path(tmp_path: Path):
    """When the entry isn't under the knowledge dir (shouldn't happen
    in real life, but be defensive), fall back to the raw path string."""
    other = tmp_path / "other.md"
    other.write_text("x", encoding="utf-8")
    out = _priming_client._wikilink(other, tmp_path / "knowledge")
    assert "other" in out


def test_shorten_truncates_with_ellipsis():
    s = _priming_client._shorten("x" * 200, max_chars=10)
    assert len(s) <= 10
    assert s.endswith("…")


def test_shorten_collapses_whitespace():
    assert _priming_client._shorten("a   b\n\nc") == "a b c"


def test_summary_empty_applies_when_str_falls_to_title(tmp_path: Path):
    """``applies-when: |`` with only blank lines → fall back to title."""
    out = _priming_client._summary(
        {"applies-when": "   \n  \n", "title": "TheTitle"}, Path("/tmp/x.md")
    )
    assert out == "TheTitle"


def test_summary_empty_applies_when_list_falls_to_title(tmp_path: Path):
    out = _priming_client._summary(
        {"applies-when": ["", "  "], "title": "TheTitle"}, Path("/tmp/x.md")
    )
    assert out == "TheTitle"


def test_summary_no_signals_falls_to_stem(tmp_path: Path):
    out = _priming_client._summary({}, Path("/tmp/the-stem.md"))
    assert "the stem" in out


def test_get_priming_no_socket_no_lib_returns_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    assert _priming_client.get_priming("python uv install") == ""


def test_get_priming_non_string_prompt_returns_empty():
    assert _priming_client.get_priming(None) == ""  # type: ignore[arg-type]
    assert _priming_client.get_priming(42) == ""  # type: ignore[arg-type]


def test_send_request_returns_none_when_socket_missing(tmp_path: Path):
    """``_send_request`` swallows transport errors and returns ``None``."""
    out = _priming_client._send_request(
        tmp_path / "nope.sock",
        {"op": "x"},
        total_budget_ms=50,
    )
    assert out is None


def test_priming_score_zero_returns_zero():
    assert _priming_client._score_doc(set(), ["a"]) == 0.0
    assert _priming_client._score_doc({"a", "b"}, []) == 0.0
    assert _priming_client._score_doc({"x"}, ["y"]) == 0.0


def test_priming_score_counts_overlapping_tokens():
    assert _priming_client._score_doc({"a", "b", "c"}, ["a", "b", "x"]) == 2.0
