"""Tests for the markdown-AST-aware wikilink extractor.

The old regex approach in scholar_prompt.check_invariants happily
matched links inside code spans, fenced code blocks, and YAML
frontmatter — producing false-positive "broken wikilink" violations.
These tests pin the new behaviour:

- Wikilinks in prose are still extracted.
- Wikilinks inside inline code (`[[…]]`) are skipped.
- Wikilinks inside fenced code blocks (```…```) are skipped.
- Wikilinks inside YAML frontmatter (---…--- at file top) are skipped.
- Aliases are preserved.
- The .md suffix on a target is normalised away.
"""
from __future__ import annotations

from agent_mem_daemon.markdown_utils import extract_wikilinks


def _targets(text: str) -> list[str]:
    return [h.target for h in extract_wikilinks(text)]


def test_extracts_plain_wikilink() -> None:
    assert _targets("See [[foo/bar]] for context.") == ["foo/bar"]


def test_extracts_alias_and_preserves_it() -> None:
    hits = extract_wikilinks("See [[foo/bar|the bar entry]] please.")
    assert len(hits) == 1
    assert hits[0].target == "foo/bar"
    assert hits[0].alias == "the bar entry"


def test_strips_md_suffix_from_target() -> None:
    assert _targets("[[foo/bar.md]]") == ["foo/bar"]


def test_skips_inline_code_span() -> None:
    # Backtick-wrapped wikilink is an example, not a link.
    text = (
        "Use `[[foo/bar]]` as the link syntax. "
        "The real link is [[foo/bar]]."
    )
    assert _targets(text) == ["foo/bar"]  # only the prose one


def test_skips_fenced_code_block() -> None:
    text = (
        "Prose link: [[foo/bar]]\n"
        "\n"
        "```\n"
        "code link: [[hidden/inside]]\n"
        "```\n"
        "\n"
        "Trailing: [[baz/qux]]\n"
    )
    assert _targets(text) == ["foo/bar", "baz/qux"]


def test_skips_indented_code_block() -> None:
    # Markdown 4-space indented = code block.
    text = (
        "Prose: [[foo/bar]]\n"
        "\n"
        "    [[indented/code]]\n"
        "\n"
        "More prose: [[baz/qux]]\n"
    )
    assert _targets(text) == ["foo/bar", "baz/qux"]


def test_skips_yaml_frontmatter() -> None:
    # Mirror the actual bug — frontmatter mentions `[[wikilinks]]` in an
    # applies-when block, and the entry body wraps it in backticks. The
    # extractor should find ZERO real links in this entry.
    text = (
        "---\n"
        "name: traversal\n"
        "applies-when: |\n"
        "  When an entry contains [[wikilinks]] to related entries\n"
        "paraphrases:\n"
        '  - "chase [[wikilinks]]"\n'
        "---\n"
        "\n"
        "# Heading\n"
        "\n"
        "Entries link via `[[wikilinks]]`.\n"
    )
    assert _targets(text) == []


def test_real_world_mixed_doc() -> None:
    # A README-like doc with one frontmatter `[[wikilinks]]` rhetorical
    # mention, two code-span examples, two real prose links, one fenced
    # code-block fake link.
    text = (
        "---\n"
        "title: Doc\n"
        "applies-when: a guide explaining the [[wikilinks]] convention\n"
        "---\n"
        "\n"
        "# Doc\n"
        "\n"
        "Use `[[path/to/entry]]` syntax. See [[projects/a/x]] for an example.\n"
        "Another inline mention: `[[also/code]]`.\n"
        "\n"
        "```\n"
        "[[code/fence/example]]\n"
        "```\n"
        "\n"
        "Final prose link: [[global/b/y]].\n"
    )
    assert _targets(text) == ["projects/a/x", "global/b/y"]


def test_multiple_links_in_one_paragraph() -> None:
    text = "Related: [[a]], [[b]], and [[c|see C]] all matter."
    hits = extract_wikilinks(text)
    assert [h.target for h in hits] == ["a", "b", "c"]
    assert hits[2].alias == "see C"


def test_empty_input_returns_empty() -> None:
    assert _targets("") == []
    assert _targets("no links here") == []


def test_link_inside_link_text_not_matched() -> None:
    # Edge case: nested brackets shouldn't generate phantom links.
    # `[[a]b]]` is technically malformed; we want it to not crash and
    # to not return a spurious `a]b`.
    text = "[[a]b]] — this is malformed"
    hits = extract_wikilinks(text)
    # The regex is forgiving but we just care that it doesn't crash.
    # If it matches something, it shouldn't contain unbalanced brackets.
    for h in hits:
        assert "[" not in h.target and "]" not in h.target
