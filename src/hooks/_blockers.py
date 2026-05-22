"""Loader + matcher for `severity: block` entries.

This module is the deterministic, no-LLM half of Tier 3 (PreToolUse
synchronous interrupt). Reading the library and matching a tool call
against the cached blocker patterns must be fast enough to run in front
of every tool invocation — target < 100 ms even with hundreds of
entries.

Design choices worth re-reading later:

- **No PyYAML dependency.** The `src/` package is the hot-path hook side
  and we don't want to drag a parser in for the few fields we actually
  need. We hand-parse the YAML frontmatter for the three fields we care
  about (``severity``, ``title``, ``block_triggers``). The trigger list
  uses a tiny indent-aware reader that recognises the exact shape
  documented in ``AGENTS.md`` — anything stranger is just skipped.

- **Cache keyed by (knowledge_dir, sentinel mtime).** ``log.md`` is the
  natural sentinel — the Scholar appends to it on every write, so its
  mtime advances whenever the library changes. If ``log.md`` is missing
  (fresh install) we fall back to the knowledge dir's own mtime, which
  ticks on dir-entry add/remove on macOS+Linux. Pre-compiled regexes
  live in the cached :class:`Trigger` so :func:`find_match` does zero
  re.compile work per call.

- **Process-local cache.** Each hook invocation spawns a fresh Python
  process, so the cache is reset for every PreToolUse anyway. The cache
  still earns its keep within a single process: tests load_blockers()
  twice in a row, and the daemon's blocker-scan invariant (if anything
  ever calls into this module from a long-lived process) gets warm
  reads.

- **Recursion guard lives in the hook, not here.** This module is
  agnostic — it never checks env vars. The caller (``pre-tool-use.py``)
  short-circuits before importing if ``CLAUDE_INVOKED_BY`` is set.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Public dataclasses ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Trigger:
    """One match rule extracted from an entry's ``block_triggers`` list.

    Two shapes are supported (documented in ``AGENTS.md``):

    - ``tool: Bash`` + ``pattern: <regex>`` — regex matched against the
      ``command`` field in ``tool_input``.
    - ``tool: Edit`` (or Write, NotebookEdit) + ``file_pattern: <regex>``
      — regex matched against ``file_path``.

    Both ``command_regex`` and ``file_path_regex`` are pre-compiled at
    load time so :func:`find_match` is a tight loop with no
    re.compile() overhead per tool call. Invalid regexes are silently
    dropped at load time (we'd rather lose one blocker than crash the
    host agent on every tool call).
    """

    tool: str
    command_regex: Optional[re.Pattern[str]] = None
    file_path_regex: Optional[re.Pattern[str]] = None


@dataclass(frozen=True)
class Blocker:
    """One library entry that opted in to pre-tool-use checking.

    ``severity`` controls the hook's response on match:
      - ``"advise"`` (default): emit ``additionalContext`` only. The
        tool proceeds; the agent gets a system reminder and decides
        for itself whether to take notice. Like a human remembering
        a relevant constraint mid-action — noticed, not paralysed.
      - ``"block"``: opt-in hard stop. Emit ``permissionDecision: deny``;
        the tool is refused outright. Reserve for genuinely dangerous
        actions (``rm -rf``, dropping prod databases).

    Carries everything the hook needs to format the message without
    re-reading the file: the absolute entry path (for the wikilink),
    the entry title, the one-line rule, and the severity level.
    """

    entry_path: Path
    title: str
    one_line_rule: str
    severity: str = "advise"
    triggers: tuple[Trigger, ...] = field(default_factory=tuple)


# ── Frontmatter parser (minimal, dependency-free) ───────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter_text, body_text)``. Empty frontmatter if
    the file doesn't start with a ``---`` fence.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end() :]


def _strip_quotes(value: str) -> str:
    """Unwrap a single layer of single or double quotes."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_simple_scalars(fm: str) -> dict[str, str]:
    """Pull out top-level `key: value` scalar pairs.

    Only catches the trivial single-line form (``key: value``). Multi-
    line / block-scalar / list values are ignored here — they're either
    handled by the dedicated ``_parse_block_triggers`` reader (for
    triggers) or simply not relevant to this module's job.
    """
    out: dict[str, str] = {}
    for line in fm.splitlines():
        # Top-level keys have no leading indent. Skip indented lines
        # (those belong to a multi-line value) and skip list-item lines.
        if not line or line[0] in (" ", "\t", "-", "#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        # Drop inline comments — naive but enough for our usage.
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[key] = _strip_quotes(value)
    return out


def _is_block_triggers_header(stripped: str, indent: int) -> bool:
    """Return True if this line is the top-level ``block_triggers:`` key."""
    stripped_r = stripped.rstrip()
    return indent == 0 and stripped_r.rstrip(":") == "block_triggers" and stripped_r.endswith(":")


def _kv_from_line(stripped: str) -> Optional[tuple[str, str]]:
    """Parse ``key: value`` out of a line; return None if malformed."""
    if ":" not in stripped:
        return None
    key, _, value = stripped.partition(":")
    key = key.strip()
    value = _strip_quotes(value.strip())
    if not key or not value:
        return None
    return key, value


def _start_list_item(stripped: str) -> dict[str, str]:
    """Build the dict for a fresh ``- key: value`` list-item line."""
    current: dict[str, str] = {}
    rest = stripped[2:].strip()
    kv = _kv_from_line(rest)
    if kv is not None:
        current[kv[0]] = kv[1]
    return current


def _block_triggers_lines(fm: str) -> list[str]:
    """Return the raw lines that sit inside the ``block_triggers:`` list.

    Stops at the first column-0 (non-empty) line after the header — that
    de-indent ends the YAML block per the AGENTS.md schema.
    """
    out: list[str] = []
    in_block = False
    for raw in fm.splitlines():
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if not in_block:
            if _is_block_triggers_header(stripped, indent):
                in_block = True
            continue
        if not stripped:
            continue
        if indent == 0:
            break
        out.append(raw)
    return out


def _parse_block_triggers(fm: str) -> list[dict[str, str]]:
    """Pull out the ``block_triggers:`` list from frontmatter.

    Recognised shape (the only shape AGENTS.md documents):

    ```yaml
    block_triggers:
      - tool: Bash
        pattern: '<regex>'
      - tool: Edit
        file_pattern: '<regex>'
    ```

    Each list item starts with a line of the form ``  - key: value`` and
    subsequent ``    key: value`` lines belong to the same item until
    the next ``-`` or until the block ends (de-indent back to column 0,
    or end of frontmatter).

    Returns a list of dicts; downstream code converts them into
    :class:`Trigger` instances. Items that fail to parse cleanly are
    skipped rather than raised — see module docstring on graceful
    degradation.
    """
    triggers: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None
    item_indent: Optional[int] = None  # column index of the `-` marker

    for raw in _block_triggers_lines(fm):
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)

        if stripped.startswith("- "):
            if current:
                triggers.append(current)
            current = _start_list_item(stripped)
            item_indent = indent
            continue
        # Continuation line of the current item.
        if current is None or item_indent is None or indent <= item_indent:
            continue
        kv = _kv_from_line(stripped)
        if kv is not None:
            current[kv[0]] = kv[1]

    if current:
        triggers.append(current)
    return triggers


def _extract_one_line_rule(body: str) -> str:
    """Pull the first non-empty, non-heading line from the body.

    Strips a leading ``#`` heading if present (the entry title is
    already in frontmatter) and trims a trailing period off so the
    rendered deny reason flows: ``... — never rm -rf /. Confirm with
    the user``. Caps at 200 chars — the deny reason becomes part of the
    agent's prompt, no need to embed a paragraph.
    """
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if len(line) > 200:
            line = line[:197] + "..."
        return line
    return ""


# ── Cache machinery ─────────────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_CACHE: dict[Path, tuple[float, list[Blocker]]] = {}


def _cache_sentinel(knowledge_dir: Path) -> float:
    """Return an mtime that ticks whenever the library changes.

    Prefers ``log.md`` because the Scholar appends to it on every write;
    falls back to the dir's own mtime; finally returns 0.0 if both are
    missing (cache always misses for an empty library, which is fine
    because the "scan" is also trivial).
    """
    log_path = knowledge_dir / "log.md"
    try:
        return log_path.stat().st_mtime
    except OSError:
        pass
    try:
        return knowledge_dir.stat().st_mtime
    except OSError:
        return 0.0


def _safe_compile(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    """Compile ``pattern`` or return None on missing/invalid regex.

    Invalid regexes are silently dropped — see module docstring on
    graceful degradation.
    """
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _trigger_from_raw(raw: dict[str, str]) -> Optional[Trigger]:
    """Turn one parsed ``block_triggers`` entry into a :class:`Trigger`.

    Returns None if there's no tool name or if both regexes are empty /
    invalid (a trigger with neither is functionally inert).
    """
    tool = raw.get("tool")
    if not tool:
        return None
    cmd_re = _safe_compile(raw.get("pattern"))
    file_re = _safe_compile(raw.get("file_pattern"))
    if cmd_re is None and file_re is None:
        return None
    return Trigger(tool=tool, command_regex=cmd_re, file_path_regex=file_re)


def _normalize_severity(raw: Optional[str]) -> str:
    """Clamp the severity field to the two values the hook understands."""
    if raw not in ("advise", "block"):
        return "advise"
    return raw


def _build_blocker(path: Path) -> Optional[Blocker]:
    """Read one entry; return a Blocker if it opts in to PreToolUse checking.

    Opt-in is now driven by the PRESENCE of ``block_triggers`` (any entry
    that lists triggers gets checked). ``severity`` controls the
    response on match:

      - ``"advise"`` (default) — emit additionalContext FYI; tool proceeds.
      - ``"block"`` — emit permissionDecision: deny; tool refused.

    Returns ``None`` for entries that have no ``block_triggers`` (the
    common case), that fail to parse, or that have zero usable triggers
    — a blocker with no triggers is functionally inert and we drop it
    rather than keep it in the cache as dead weight.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fm, body = _split_frontmatter(text)
    if not fm:
        return None

    raw_triggers = _parse_block_triggers(fm)
    if not raw_triggers:
        return None

    triggers = tuple(t for t in (_trigger_from_raw(r) for r in raw_triggers) if t is not None)
    if not triggers:
        return None

    scalars = _parse_simple_scalars(fm)
    return Blocker(
        entry_path=path.resolve(),
        title=scalars.get("title", path.stem),
        one_line_rule=_extract_one_line_rule(body),
        severity=_normalize_severity(scalars.get("severity")),
        triggers=triggers,
    )


def _scan_blockers(knowledge_dir: Path) -> list[Blocker]:
    """Walk the knowledge tree once, collecting every blocker entry.

    Skips ``_archive/`` (archived entries shouldn't fire interrupts) and
    skips the top-level ``index.md`` / ``log.md`` files.
    """
    out: list[Blocker] = []
    if not knowledge_dir.exists():
        return out
    for path in knowledge_dir.rglob("*.md"):
        try:
            rel = path.relative_to(knowledge_dir)
        except ValueError:
            continue
        # Drop archived entries and top-level admin files.
        parts = rel.parts
        if parts and parts[0] == "_archive":
            continue
        if rel.name in ("index.md", "log.md", "README.md"):
            continue
        blocker = _build_blocker(path)
        if blocker is not None:
            out.append(blocker)
    return out


# ── Public API ──────────────────────────────────────────────────────────


def load_blockers(knowledge_dir: Path) -> list[Blocker]:
    """Return every entry under ``knowledge_dir`` that opts in to
    blocking.

    Cached per ``(knowledge_dir, sentinel_mtime)``. The sentinel mtime
    advances whenever ``log.md`` is touched (Scholar writes append to
    it); if ``log.md`` is missing we fall back to the directory's own
    mtime. Each call pays exactly one stat() on the sentinel and a
    full rescan only if the mtime moved.

    Thread-safe under a single module-level lock — multiple hooks can
    call this concurrently within the same process (rare but possible
    in the daemon priming module).
    """
    key = knowledge_dir.resolve()
    sentinel = _cache_sentinel(key)

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == sentinel:
            return cached[1]
        blockers = _scan_blockers(key)
        _CACHE[key] = (sentinel, blockers)
        return blockers


def clear_cache() -> None:
    """Drop all cached blockers. Test-only — production hooks run in
    fresh subprocesses so the cache is naturally cold per call.
    """
    with _CACHE_LOCK:
        _CACHE.clear()


def _trigger_fires(trigger: Trigger, tool_name: str, command: str, file_path: str) -> bool:
    """Return True if this single trigger matches the incoming tool call."""
    if trigger.tool != tool_name:
        return False
    if trigger.command_regex is not None and command and trigger.command_regex.search(command):
        return True
    if (
        trigger.file_path_regex is not None
        and file_path
        and trigger.file_path_regex.search(file_path)
    ):
        return True
    return False


def find_match(
    blockers: list[Blocker],
    tool_name: str,
    tool_input: dict[str, Any],
) -> Optional[Blocker]:
    """Return the first blocker whose trigger matches this tool call.

    Matching rules:

    - A trigger fires only when ``trigger.tool`` equals ``tool_name``
      (case-sensitive; Claude Code's tool names are PascalCase like
      ``Bash``, ``Edit``, ``Write``, ``NotebookEdit``).
    - Bash triggers (``pattern``) match against the ``command`` field
      in ``tool_input`` using ``re.search`` (substring-style, anchored
      with ``^`` / ``$`` only when the rule author wrote them).
    - Edit/Write/NotebookEdit triggers (``file_pattern``) match against
      ``file_path``.

    Returns the first match in iteration order so the rule author can
    rely on stable behaviour when two blockers might both fire.
    """
    if not blockers:
        return None

    command_val: Any = tool_input.get("command")
    file_path_val: Any = tool_input.get("file_path")
    command = command_val if isinstance(command_val, str) else ""
    file_path = file_path_val if isinstance(file_path_val, str) else ""

    for blocker in blockers:
        if any(_trigger_fires(t, tool_name, command, file_path) for t in blocker.triggers):
            return blocker
    return None


def rel_to_knowledge(entry_path: Path, knowledge_dir: Optional[Path] = None) -> str:
    """Format the entry path as a wikilink target.

    If ``knowledge_dir`` is provided and the entry sits under it, return
    the path relative to ``knowledge/`` without the ``.md`` extension
    (matching the wikilink convention from ``AGENTS.md`` §8). Otherwise
    fall back to the bare filename stem so the deny reason still names
    *something*.
    """
    if knowledge_dir is not None:
        try:
            rel = entry_path.resolve().relative_to(knowledge_dir.resolve())
            return str(rel).removesuffix(".md")
        except ValueError:
            pass
    # Best-effort: try to find a "knowledge" segment in the path and
    # slice from there. Works for the default ~/.agent-mem/knowledge/...
    # layout without needing the caller to thread the dir through.
    parts = entry_path.parts
    for i, part in enumerate(parts):
        if part == "knowledge" and i + 1 < len(parts):
            tail = os.sep.join(parts[i + 1 :])
            return tail.removesuffix(".md")
    return entry_path.stem
