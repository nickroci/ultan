"""In-process structural lint for the knowledge library.

Powers the ``Lint (structural-only)`` section of ``agent-mem doctor``. Three
graph checks over the live ``knowledge/`` tree:

  - broken wikilinks  (error)      — ``[[link]]`` with no backing entry/folder
  - orphan pages      (warning)    — a content entry nothing links to
  - sparse articles   (suggestion) — fewer than ``SPARSE_MIN_WORDS`` words

Pure-stdlib; no LLM, no subprocess. Ported from the retired
``src/scripts/lint.py --structural-only`` with daemon-era corrections:

1. It scans the live free-form ``knowledge/`` tree via the caller's entry
   iterator (the same selector BM25 indexes), not the obsolete
   ``concepts/connections/qa`` sub-tiers the old script hard-coded.
2. Wikilink resolution MIRRORS the daemon's ``_validation.wikilink_resolves``
   (tools/search can't import the daemon — it's the other way round): a trailing
   ``/`` is a folder link resolving to ``<link>/README.md``; ``_archive/`` and
   ``daily/`` targets are always valid; resolution is root-relative first, then
   sibling-relative to the linking file's directory. Without this, the READMEs'
   ``[[subdir/]]`` listings all looked "broken". Wikilink extraction also skips
   code spans / fences / frontmatter so example links in docs don't false-flag.
3. READMEs are auto-maintained directory indexes, not content, so they are
   skipped by the orphan/sparse checks.

The old script's missing-backlink check is intentionally dropped: this library
links related content one-directionally (the daemon only enforces parent/child
backlinks structurally), so it was pure noise. The opt-in LLM contradiction
sweep is likewise not ported. Exit-code parity: :func:`run_structural_lint`
returns ``rc == 1`` iff there is at least one error-severity issue (a broken
link); warnings/suggestions never trip it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# Regions where a ``[[…]]`` is an example, not a real link. Stripped before
# wikilink extraction — a dependency-free approximation of the daemon's
# markdown-aware extractor (which uses markdown-it-py). Without this, docs that
# discuss wikilinks (``​`[[wikilink]]`​``) get false broken-link errors.
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
SPARSE_MIN_WORDS = 200


@dataclass(frozen=True)
class LintIssue:
    """One structural finding. ``severity`` is ``error`` | ``warning`` |
    ``suggestion``; only ``error`` (a broken link) affects the exit code."""

    severity: str
    check: str
    file: str
    detail: str


def _prose(content: str) -> str:
    """Drop leading frontmatter, fenced code blocks, and inline code spans so
    wikilinks shown as examples in code aren't mistaken for real links."""
    content = _FRONTMATTER_RE.sub("", content, count=1)
    content = _FENCE_RE.sub("", content)
    return _INLINE_CODE_RE.sub("", content)


def _wikilinks(content: str) -> List[str]:
    """Wikilink targets in the PROSE of ``content`` (code/frontmatter skipped),
    with any ``|alias`` / ``#anchor`` stripped."""
    out: List[str] = []
    for raw in _WIKILINK_RE.findall(_prose(content)):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target:
            out.append(target)
    return out


def _is_external(link: str) -> bool:
    """``_archive/`` and ``daily/`` targets are valid but outside the active
    graph (parity with the daemon's ``wikilink_resolves``)."""
    return link.startswith(("_archive/", "daily/")) or "/_archive/" in link


def _rel_for(link: str) -> str:
    """The knowledge-relative ``.md`` path a wikilink names: folder links
    (trailing ``/``) point at ``<link>/README.md``."""
    if link.endswith("/"):
        return f"{link}README.md"
    return link if link.endswith(".md") else f"{link}.md"


def _resolve(link: str, parent_dir: Path, knowledge_dir: Path) -> Optional[Path]:
    """The file a (non-external) wikilink resolves to, or ``None`` if it
    doesn't. Root-relative first, then sibling-relative to ``parent_dir`` —
    mirrors ``_validation.wikilink_resolves``."""
    rel = _rel_for(link)
    root = knowledge_dir / rel
    if root.exists():
        return root
    sibling = parent_dir / rel
    return sibling if sibling.exists() else None


def _target_of(entry: Path, knowledge_dir: Path) -> str:
    """The canonical id for an entry: its knowledge-relative path, no ``.md``."""
    return entry.relative_to(knowledge_dir).with_suffix("").as_posix()


def _scan_links(entry: Path, knowledge_dir: Path, text: str) -> Tuple[List[str], Set[str]]:
    """Return ``(broken_links, resolved_target_ids)`` for one entry.

    ``broken_links`` are the raw link strings that don't resolve (excluding
    external archive/daily). ``resolved_target_ids`` are the canonical ids of
    the in-tree entries this one points at (for the inbound graph).
    """
    broken: List[str] = []
    resolved: Set[str] = set()
    for link in _wikilinks(text):
        if _is_external(link):
            continue
        target = _resolve(link, entry.parent, knowledge_dir)
        if target is None:
            broken.append(link)
        else:
            resolved.add(_target_of(target, knowledge_dir))
    return broken, resolved


def _word_count(content: str) -> int:
    """Word count excluding a leading YAML frontmatter block."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3 :]
    return len(content.split())


def collect_issues(knowledge_dir: Path, entries: Iterable[Path]) -> List[LintIssue]:
    """Run the structural checks over ``entries`` (paths under ``knowledge_dir``)."""
    entry_list = sorted(set(entries))
    texts = {e: _safe_read(e) for e in entry_list}

    # Resolved link graph: broken links per entry and inbound counts. Built
    # once so the orphan check is accurate w.r.t. folder links and
    # sibling-relative resolution.
    broken_of: Dict[Path, List[str]] = {}
    inbound: Dict[str, int] = {_target_of(e, knowledge_dir): 0 for e in entry_list}
    for entry in entry_list:
        broken, resolved = _scan_links(entry, knowledge_dir, texts[entry])
        broken_of[entry] = broken
        for tid in resolved:
            if tid in inbound:
                inbound[tid] += 1

    issues: List[LintIssue] = []
    for entry in entry_list:
        rel = entry.relative_to(knowledge_dir).as_posix()
        src = _target_of(entry, knowledge_dir)
        for link in broken_of[entry]:
            issues.append(
                LintIssue("error", "broken_link", rel, f"[[{link}]] - target does not exist")
            )
        if entry.name != "README.md":  # READMEs are indexes, not content
            issues.extend(_content_issues(rel, src, texts[entry], inbound))
    return issues


def _safe_read(entry: Path) -> str:
    try:
        return entry.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _content_issues(rel: str, src: str, text: str, inbound: Dict[str, int]) -> List[LintIssue]:
    """Orphan (warning) and sparse (suggestion) findings for one content entry."""
    issues: List[LintIssue] = []
    if inbound.get(src, 0) == 0:
        issues.append(
            LintIssue("warning", "orphan_page", rel, f"no other entry links to [[{src}]]")
        )
    words = _word_count(text)
    if words < SPARSE_MIN_WORDS:
        issues.append(
            LintIssue(
                "suggestion",
                "sparse_article",
                rel,
                f"{words} words (min recommended: {SPARSE_MIN_WORDS})",
            )
        )
    return issues


def run_structural_lint(
    knowledge_dir: Path, entries: Iterable[Path], *, max_examples: int = 10
) -> Tuple[int, str]:
    """Run the four checks and render a bounded report.

    Returns ``(rc, report)`` where ``rc == 1`` iff any error-severity issue
    (a broken link) exists — parity with the retired ``lint.py``. ``report``
    shows up to ``max_examples`` examples per check so the output stays readable
    on a large library.
    """
    issues = collect_issues(knowledge_dir, entries)
    if not issues:
        return 0, "All structural checks passed."

    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    suggestions = sum(1 for i in issues if i.severity == "suggestion")
    lines = [
        f"{len(issues)} issue(s): {errors} error, {warnings} warning, {suggestions} suggestion",
    ]
    by_check: Dict[str, List[LintIssue]] = {}
    for issue in issues:
        by_check.setdefault(issue.check, []).append(issue)
    for check, items in sorted(by_check.items()):
        lines.append(f"  {check}: {len(items)}")
        for issue in items[:max_examples]:
            lines.append(f"    [{issue.severity[0]}] {issue.file}: {issue.detail}")
        if len(items) > max_examples:
            lines.append(f"    … +{len(items) - max_examples} more")
    return (1 if errors else 0), "\n".join(lines)
