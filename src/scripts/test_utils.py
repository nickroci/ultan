"""Behavioural regression tests for ``scripts/utils.py``.

Pure / file-IO helpers only. Every test isolates via the ``store_home``
fixture (``AGENT_MEM_HOME`` pinned to a tmp dir), so reads/writes land in
a throwaway store and never touch the user's real ``~/.agent-mem``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import utils

# ── State round-trip ─────────────────────────────────────────────────


def test_load_state_returns_defaults_when_absent(store_home: Path) -> None:
    state = utils.load_state()
    assert state == {
        "ingested": {},
        "query_count": 0,
        "last_lint": None,
        "total_cost": 0.0,
    }


def test_save_then_load_state_round_trips(store_home: Path) -> None:
    written = {
        "ingested": {"2026-05-28.md": {"hash": "abc123", "compiled_at": "t", "cost_usd": 0.5}},
        "query_count": 7,
        "last_lint": "2026-05-28T00:00:00",
        "total_cost": 1.25,
    }
    utils.save_state(written)  # type: ignore[arg-type]
    # save_state creates the state dir as a side effect.
    assert (store_home / "state" / "state.json").exists()
    assert utils.load_state() == written


# ── slugify ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello World", "hello-world"),
        ("  Trim Me  ", "trim-me"),
        ("Foo/Bar: Baz!", "foobar-baz"),
        ("multiple   spaces__and_underscores", "multiple-spaces-and-underscores"),
        ("--leading-and-trailing--", "leading-and-trailing"),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert utils.slugify(raw) == expected


# ── extract_wikilinks ────────────────────────────────────────────────


def test_extract_wikilinks_finds_all() -> None:
    content = "See [[global/concepts/a]] and [[concepts/b]] but not [single]."
    assert utils.extract_wikilinks(content) == ["global/concepts/a", "concepts/b"]


def test_extract_wikilinks_empty_when_none() -> None:
    assert utils.extract_wikilinks("no links here") == []


# ── wiki_article_exists ──────────────────────────────────────────────


def test_wiki_article_exists_new_form(store_home: Path) -> None:
    target = store_home / "knowledge" / "global" / "concepts" / "x.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# x\n", encoding="utf-8")
    assert utils.wiki_article_exists("global/concepts/x") is True


def test_wiki_article_exists_legacy_fallback(store_home: Path) -> None:
    # On disk under global/, but referenced by the legacy short form.
    target = store_home / "knowledge" / "global" / "concepts" / "y.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# y\n", encoding="utf-8")
    assert utils.wiki_article_exists("concepts/y") is True


def test_wiki_article_exists_missing(store_home: Path) -> None:
    assert utils.wiki_article_exists("global/concepts/nope") is False


# ── read_wiki_index ──────────────────────────────────────────────────


def test_read_wiki_index_default_when_absent(store_home: Path) -> None:
    out = utils.read_wiki_index()
    assert out.startswith("# Knowledge Base Index")
    assert "| Article | Summary |" in out


def test_read_wiki_index_reads_existing(store_home: Path) -> None:
    index = store_home / "knowledge" / "index.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("# Custom Index\n", encoding="utf-8")
    assert utils.read_wiki_index() == "# Custom Index\n"


# ── list_raw_files ───────────────────────────────────────────────────


def test_list_raw_files_empty_when_no_dir(store_home: Path) -> None:
    assert utils.list_raw_files() == []


def test_list_raw_files_sorted(store_home: Path) -> None:
    daily = store_home / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for name in ("2026-05-02.md", "2026-05-01.md"):
        (daily / name).write_text("log\n", encoding="utf-8")
    (daily / "notes.txt").write_text("ignore me\n", encoding="utf-8")
    names = [p.name for p in utils.list_raw_files()]
    assert names == ["2026-05-01.md", "2026-05-02.md"]


# ── count_inbound_links ──────────────────────────────────────────────


def test_count_inbound_links_new_and_legacy_forms(
    store_home: Path, article_writer: Callable[..., Path]
) -> None:
    concepts = store_home / "knowledge" / "global" / "concepts"
    target = "global/concepts/hub"
    article_writer(concepts / "hub.md", title="Hub", body="central")
    # one article links via the new full form...
    article_writer(concepts / "a.md", title="A", body="x", links=["global/concepts/hub"])
    # ...another via the legacy short form (still counts as inbound).
    article_writer(concepts / "b.md", title="B", body="y", links=["concepts/hub"])
    # and one that links elsewhere (should not count).
    article_writer(concepts / "c.md", title="C", body="z", links=["global/concepts/other"])
    assert utils.count_inbound_links(target) == 2


def test_count_inbound_links_excludes_self(
    store_home: Path, article_writer: Callable[..., Path]
) -> None:
    concepts = store_home / "knowledge" / "global" / "concepts"
    self_path = article_writer(
        concepts / "selfref.md", title="Self", body="x", links=["global/concepts/selfref"]
    )
    assert utils.count_inbound_links("global/concepts/selfref", exclude_file=self_path) == 0


# ── get_article_word_count ───────────────────────────────────────────


def test_get_article_word_count_strips_frontmatter(store_home: Path) -> None:
    path = store_home / "art.md"
    path.write_text(
        "---\nid: art\ntitle: Skip This Frontmatter\n---\n\none two three four\n",
        encoding="utf-8",
    )
    assert utils.get_article_word_count(path) == 4


def test_get_article_word_count_no_frontmatter(store_home: Path) -> None:
    path = store_home / "plain.md"
    path.write_text("alpha beta gamma", encoding="utf-8")
    assert utils.get_article_word_count(path) == 3


# ── build_index_entry ────────────────────────────────────────────────


def test_build_index_entry_strips_md_and_wraps_link() -> None:
    row = utils.build_index_entry(
        "global/concepts/foo.md", "A summary", "daily/2026-05-28.md", "2026-05-28"
    )
    assert row == "| [[global/concepts/foo]] | A summary | daily/2026-05-28.md | 2026-05-28 |"
