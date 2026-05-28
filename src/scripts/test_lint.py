"""Behavioural regression tests for ``scripts/lint.py``.

Exercises the structural (non-LLM) checks against a tiny seeded knowledge
dir under ``AGENT_MEM_HOME``. The ``check_contradictions`` /
``run_*`` LLM paths are not touched — they call the SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import lint


def _details(issues: list[lint.Issue]) -> list[str]:
    return [i["detail"] for i in issues]


# ── check_broken_links ───────────────────────────────────────────────


def test_check_broken_links_flags_only_missing(
    store_home: Path, article_writer: Callable[..., Path]
) -> None:
    concepts = store_home / "knowledge" / "global" / "concepts"
    article_writer(concepts / "real.md", title="Real", body="exists")
    article_writer(
        concepts / "src.md",
        title="Src",
        body="links out",
        # one valid target, one broken, one daily-log ref (always valid).
        links=["global/concepts/real", "global/concepts/ghost", "daily/2026-05-28.md"],
    )
    issues = lint.check_broken_links()
    assert [i["check"] for i in issues] == ["broken_link"]
    assert "global/concepts/ghost" in issues[0]["detail"]
    assert issues[0]["severity"] == "error"


# ── check_orphan_pages ───────────────────────────────────────────────


def test_check_orphan_pages_flags_unlinked(
    store_home: Path, article_writer: Callable[..., Path]
) -> None:
    concepts = store_home / "knowledge" / "global" / "concepts"
    # linked points at hub; hub has an inbound link; lonely has none.
    article_writer(concepts / "hub.md", title="Hub", body="central")
    article_writer(concepts / "linked.md", title="Linked", body="x", links=["global/concepts/hub"])
    article_writer(concepts / "lonely.md", title="Lonely", body="nobody links here")

    issues = lint.check_orphan_pages()
    orphan_files = {i["file"] for i in issues}
    # `linked` and `lonely` have zero inbound links; `hub` has one.
    assert "global/concepts/lonely.md" in orphan_files
    assert "global/concepts/hub.md" not in orphan_files
    assert all(i["severity"] == "warning" for i in issues)


# ── check_missing_backlinks ──────────────────────────────────────────


def test_check_missing_backlinks_flags_asymmetric(
    store_home: Path, article_writer: Callable[..., Path]
) -> None:
    concepts = store_home / "knowledge" / "global" / "concepts"
    # a -> b, but b does not link back to a.
    article_writer(concepts / "a.md", title="A", body="x", links=["global/concepts/b"])
    article_writer(concepts / "b.md", title="B", body="y")

    issues = lint.check_missing_backlinks()
    assert len(issues) == 1
    assert issues[0]["check"] == "missing_backlink"
    assert issues[0]["auto_fixable"] is True
    assert "global/concepts/a" in issues[0]["detail"]


def test_check_missing_backlinks_symmetric_is_clean(
    store_home: Path, article_writer: Callable[..., Path]
) -> None:
    concepts = store_home / "knowledge" / "global" / "concepts"
    article_writer(concepts / "a.md", title="A", body="x", links=["global/concepts/b"])
    article_writer(concepts / "b.md", title="B", body="y", links=["global/concepts/a"])
    assert lint.check_missing_backlinks() == []


# ── check_sparse_articles ────────────────────────────────────────────


def test_check_sparse_articles_flags_short(
    store_home: Path, article_writer: Callable[..., Path]
) -> None:
    concepts = store_home / "knowledge" / "global" / "concepts"
    article_writer(concepts / "thin.md", title="Thin", body="just a few words")
    article_writer(
        concepts / "fat.md",
        title="Fat",
        body=" ".join(f"word{n}" for n in range(250)),
    )
    issues = lint.check_sparse_articles()
    files = {i["file"] for i in issues}
    assert "global/concepts/thin.md" in files
    assert "global/concepts/fat.md" not in files
    assert all(i["severity"] == "suggestion" for i in issues)


# ── generate_report ──────────────────────────────────────────────────


def test_generate_report_groups_by_severity() -> None:
    issues: list[lint.Issue] = [
        lint.Issue(severity="error", check="broken_link", file="a.md", detail="bad link"),
        lint.Issue(severity="warning", check="orphan_page", file="b.md", detail="orphan"),
        lint.Issue(
            severity="suggestion",
            check="missing_backlink",
            file="c.md",
            detail="needs backlink",
            auto_fixable=True,
        ),
    ]
    report = lint.generate_report(issues)
    assert "**Total issues:** 3" in report
    assert "- Errors: 1" in report
    assert "## Errors" in report and "## Warnings" in report and "## Suggestions" in report
    assert "bad link" in report
    assert "(auto-fixable)" in report


def test_generate_report_healthy_when_empty() -> None:
    report = lint.generate_report([])
    assert "**Total issues:** 0" in report
    assert "All checks passed. Knowledge base is healthy." in report
