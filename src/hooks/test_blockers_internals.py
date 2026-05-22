"""Cover the corners of ``_blockers`` that the end-to-end hook tests
miss: the wikilink formatter, the frontmatter parsers, and the cache-
sentinel fallbacks."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import dedent

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import _blockers  # noqa: E402


def test_rel_to_knowledge_with_explicit_dir(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    entry = knowledge / "global" / "concepts" / "a.md"
    entry.parent.mkdir(parents=True)
    entry.write_text("x", encoding="utf-8")
    out = _blockers.rel_to_knowledge(entry, knowledge)
    assert out == os.sep.join(["global", "concepts", "a"])


def test_rel_to_knowledge_outside_knowledge_dir(tmp_path: Path):
    """Entry not under the provided knowledge dir → fallback search
    for any segment named ``knowledge``."""
    entry = tmp_path / "knowledge" / "p" / "e.md"
    entry.parent.mkdir(parents=True)
    entry.write_text("x", encoding="utf-8")
    # Pass a different "knowledge" dir => relative_to raises => fallback.
    out = _blockers.rel_to_knowledge(entry, tmp_path / "elsewhere")
    assert out.endswith(os.sep.join(["p", "e"])) or out.endswith("p/e")


def test_rel_to_knowledge_no_segment_falls_back_to_stem(tmp_path: Path):
    entry = tmp_path / "no-knowledge-here" / "entry.md"
    entry.parent.mkdir(parents=True)
    entry.write_text("x", encoding="utf-8")
    out = _blockers.rel_to_knowledge(entry, None)
    assert out == "entry"


def test_rel_to_knowledge_no_dir_uses_path_search(tmp_path: Path):
    """When knowledge_dir is None, the helper walks the path parts to
    find a ``knowledge`` segment."""
    entry = tmp_path / "knowledge" / "p" / "e.md"
    entry.parent.mkdir(parents=True)
    entry.write_text("x", encoding="utf-8")
    out = _blockers.rel_to_knowledge(entry, None)
    assert out.endswith("e")
    assert "p" in out


def test_cache_sentinel_falls_back_to_dir_mtime(tmp_path: Path):
    """No log.md → use the knowledge dir's own mtime."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    mtime = _blockers._cache_sentinel(knowledge)
    # Knowledge dir exists so we get its mtime, not 0.
    assert mtime > 0


def test_cache_sentinel_missing_dir_returns_zero(tmp_path: Path):
    out = _blockers._cache_sentinel(tmp_path / "missing")
    assert out == 0.0


def test_find_match_with_non_dict_tool_input():
    """find_match accepts only dict tool_input."""
    out = _blockers.find_match([], "Bash", "not a dict")  # type: ignore[arg-type]
    assert out is None


def test_find_match_with_empty_blockers():
    assert _blockers.find_match([], "Bash", {"command": "rm -rf"}) is None


def test_parse_block_triggers_handles_de_indent(tmp_path: Path):
    """De-indenting back to column 0 mid-block ends the trigger list."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "log.md").write_text("# Log", encoding="utf-8")
    (knowledge / "e.md").write_text(
        dedent(
            """
            ---
            severity: block
            title: "t"
            block_triggers:
              - tool: Bash
                pattern: 'foo'
            other_top_level: x
            ---

            Body
            """
        ).lstrip(),
        encoding="utf-8",
    )
    blockers = _blockers.load_blockers(knowledge)
    assert len(blockers) == 1
    assert blockers[0].triggers[0].tool == "Bash"


def test_parse_block_triggers_continuation_line():
    """Multi-line trigger items: subsequent indented lines belong to
    the current item."""
    frontmatter = dedent(
        """
        block_triggers:
          - tool: Bash
            pattern: 'foo'
          - tool: Edit
            file_pattern: 'x'
        """
    ).strip()
    out = _blockers._parse_block_triggers(frontmatter)
    assert len(out) == 2
    assert out[0]["tool"] == "Bash"
    assert out[1]["tool"] == "Edit"


def test_parse_block_triggers_skips_blank_lines():
    frontmatter = dedent(
        """
        block_triggers:
          - tool: Bash
            pattern: 'foo'

          - tool: Edit
            file_pattern: 'x'
        """
    ).strip()
    out = _blockers._parse_block_triggers(frontmatter)
    assert len(out) == 2


def test_parse_block_triggers_no_block_returns_empty():
    """Frontmatter that doesn't include block_triggers: → empty list."""
    out = _blockers._parse_block_triggers("severity: block\ntitle: x")
    assert out == []


def test_parse_simple_scalars_strips_inline_comments():
    out = _blockers._parse_simple_scalars("title: foo # comment here\nseverity: block")
    assert out["title"] == "foo"


def test_parse_simple_scalars_strips_quotes():
    out = _blockers._parse_simple_scalars('title: "quoted value"')
    assert out["title"] == "quoted value"


def test_split_frontmatter_no_fence_returns_whole_text():
    fm, body = _blockers._split_frontmatter("# Just a heading\nbody")
    assert fm == ""
    assert body == "# Just a heading\nbody"


def test_extract_one_line_rule_skips_headings():
    out = _blockers._extract_one_line_rule("# heading\n\nfirst real line\n\nsecond")
    assert out == "first real line"


def test_extract_one_line_rule_caps_at_200_chars():
    out = _blockers._extract_one_line_rule("a" * 500)
    assert len(out) == 200
    assert out.endswith("...")


def test_extract_one_line_rule_returns_empty_when_nothing_usable():
    out = _blockers._extract_one_line_rule("\n# h1\n\n# h2\n")
    assert out == ""


def test_build_blocker_unreadable_returns_none(tmp_path: Path, monkeypatch):
    """File whose read_text raises => skipped."""
    entry = tmp_path / "x.md"
    entry.write_text("--- \nseverity: block\n", encoding="utf-8")

    def boom(self, *a, **k):
        raise OSError("denied")

    monkeypatch.setattr(Path, "read_text", boom)
    assert _blockers._build_blocker(entry) is None


def test_clear_cache_drops_all(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "log.md").write_text("x", encoding="utf-8")
    _blockers.load_blockers(knowledge)
    _blockers.clear_cache()
    # After clear, cache key is gone — second load_blockers triggers a scan.
    assert knowledge.resolve() not in _blockers._CACHE
