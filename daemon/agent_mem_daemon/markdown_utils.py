"""Markdown-aware helpers for the daemon.

Until now the daemon used a single regex (``\\[\\[([^\\]]+?)\\]\\]``) to find
wikilinks anywhere in a markdown file. That happily matches code samples
like ``\\`[[wikilinks]]\\``` and YAML-frontmatter prose, producing false-
positive "broken wikilink" violations.

This module parses each file with markdown-it-py and only returns
wikilinks that live in prose — inline code spans and fenced/indented
code blocks are skipped, and YAML frontmatter (the ``--- ... ---`` block
at the top of every entry) is stripped before parsing.

Used by:
  - scholar_prompt.check_invariants (post-write validator)
  - llm._make_path_guard (pre-write Scholar guard, when added)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from markdown_it import MarkdownIt

# Matches a wikilink target + optional alias.
# Group 1 = target text, Group 2 = "|alias" suffix (may be missing).
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\|]+?)(\|[^\[\]]*)?\]\]")

# YAML frontmatter — only at the very start of the file. Non-greedy so
# inline `---` rules later in the body don't get swallowed.
_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


@dataclass(frozen=True)
class WikilinkHit:
    """A wikilink found in prose (non-code) markdown."""

    target: str  # the link target, with .md stripped (e.g. "global/user/foo")
    alias: str | None  # the alias if `[[target|alias]]` was used, else None
    raw: str  # the original `[[…]]` substring as it appeared


def _strip_frontmatter(text: str) -> str:
    """Drop a leading `---\\n...\\n---\\n` YAML block if present."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def _normalize_target(raw_target: str) -> str:
    """Strip whitespace and an optional trailing ``.md`` from a link target."""
    t = raw_target.strip()
    if t.endswith(".md"):
        t = t[:-3]
    return t


def _iter_prose_text(text: str) -> Iterable[str]:
    """Yield text from markdown nodes that aren't code spans or code
    blocks. Frontmatter is stripped before parsing.

    Implementation note: we use markdown-it-py's token stream rather than
    walking a full AST. The token stream is flat — block tokens (paragraph,
    heading, fence, code_block, etc.) appear at the top level, and inline
    content is nested under ``inline`` tokens via ``.children``. Skipping
    a code fence is just "ignore tokens of type fence/code_block"; skipping
    an inline code span is "ignore children of type code_inline".
    """
    md = MarkdownIt("commonmark")
    body = _strip_frontmatter(text)
    tokens = md.parse(body)

    for tok in tokens:
        ttype = tok.type
        # Block-level code — fenced (```...```) and indented (4-space).
        if ttype in ("fence", "code_block"):
            continue
        # html_block could legally contain `[[…]]` in HTML comments etc.
        # Most repo content doesn't use raw HTML; skip to stay safe.
        if ttype == "html_block":
            continue
        if ttype != "inline":
            continue
        # Inline token — walk its children, collecting only text nodes.
        children = tok.children or []
        for child in children:
            ct = child.type
            if ct in ("code_inline", "html_inline"):
                continue
            # text, softbreak, hardbreak, link_open, link_close, em_open,
            # em_close, strong_*, image, etc. — emit their .content where
            # it's set.
            content = getattr(child, "content", "") or ""
            if content:
                yield content


def extract_wikilinks(text: str) -> List[WikilinkHit]:
    """Return every wikilink that appears in prose (not code, not
    frontmatter).

    Aliases are preserved. ``.md`` suffixes on the target are stripped so
    downstream code can compare against canonical link paths.
    """
    out: List[WikilinkHit] = []
    for chunk in _iter_prose_text(text):
        for m in _WIKILINK_RE.finditer(chunk):
            raw_target = m.group(1)
            alias_with_pipe = m.group(2) or ""
            alias = alias_with_pipe[1:] if alias_with_pipe else None
            out.append(
                WikilinkHit(
                    target=_normalize_target(raw_target),
                    alias=alias,
                    raw=m.group(0),
                )
            )
    return out
