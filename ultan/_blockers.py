"""Deterministic PreToolUse blocker check for the `ultan` plugin path.

This is the no-LLM half of Tier 3 (PreToolUse synchronous interrupt). It is
the only synchronous point Claude Code exposes: the hook fires **before** a
tool runs, and a ``permissionDecision: deny`` response refuses the call
outright. The async nudge pipeline can't help here — by the time a "never
deploy to prod" nudge lands, the destructive ``gcloud deploy`` has already
run.

Ported from the legacy ``src/hooks/_blockers.py`` (load + match + cache) plus
the decision-formatting that used to live in ``src/hooks/pre_tool_use.py``.
The two halves are merged so the Phase-2 integrator can call a single
:func:`evaluate` and get back a ready-to-emit ``hookSpecificOutput`` extra
dict — no message-formatting logic needs to live in ``_hooks.py``.

Hot-path invariant (see ``_hooks.py`` / ``_events.py`` docstrings): this
module must stay FAST and stdlib-only — NO torch / sentence-transformers /
heavy imports. It is pure stdlib, which fits; ``tests/test_hook_import.py``
guards the invariant once Phase 2 wires it in.

Design choices worth re-reading later (carried over from legacy):

- **No PyYAML dependency.** We hand-parse the YAML frontmatter for the three
  fields we need (``severity``, ``title``, ``block_triggers``). The trigger
  list uses a tiny indent-aware reader; anything stranger is skipped.

- **Cache keyed by (knowledge_dir, sentinel mtime).** ``log.md`` is the
  natural sentinel — the Scholar appends to it on every write, so its mtime
  advances whenever the library changes. If it's missing we fall back to the
  knowledge dir's own mtime. Regexes are pre-compiled into the cached
  :class:`Trigger` so :func:`find_match` does zero ``re.compile`` per call.

- **Opt-in = presence of ``block_triggers``.** Any entry that lists triggers
  gets checked; ``severity`` (clamped to advise|block, default advise)
  controls the response on match.

- **Knowledge-dir resolution via the daemon path**, not legacy ``config.py``:
  ``${AGENT_MEM_HOME:-~/.agent-mem}/knowledge`` (see :func:`knowledge_dir`),
  matching ``_priming._knowledge_dir`` and ``_events._home``.

- **Recursion guard lives in the caller.** When the daemon spawns the
  Scholar/Librarian as Agent-SDK subprocesses it sets ``CLAUDE_INVOKED_BY``;
  re-entering the blocker check there would block the Scholar's own
  Edit/Write calls and deadlock the daemon. :func:`evaluate` short-circuits
  on that env var so the Phase-2 dispatch branch doesn't have to remember to.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ._daemon import _home  # pyright: ignore[reportPrivateUsage]  # intra-package

# ── Path resolution (daemon path, not legacy config.py) ──────────────────


def knowledge_dir() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}/knowledge``.

    Mirrors ``_priming._knowledge_dir`` and routes through ``_daemon._home``
    so this module carries no dependency on the heavy ``src/scripts/config``.
    """
    return _home() / "knowledge"


# ── Public dataclasses ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Trigger:
    """One match rule extracted from an entry's ``block_triggers`` list.

    Two shapes are supported (documented in ``AGENTS.md``):

    - ``tool: Bash`` + ``pattern: <regex>`` — regex matched against the
      ``command`` field in ``tool_input``.
    - ``tool: Edit`` (or Write, NotebookEdit) + ``file_pattern: <regex>`` —
      regex matched against ``file_path``.

    Both regexes are pre-compiled at load time so :func:`find_match` is a tight
    loop with no ``re.compile`` per tool call. Invalid regexes are silently
    dropped at load time (we'd rather lose one blocker than crash the host
    agent on every tool call).
    """

    tool: str
    command_regex: Optional[re.Pattern[str]] = None
    file_path_regex: Optional[re.Pattern[str]] = None


@dataclass(frozen=True)
class Blocker:
    """One library entry that opted in to pre-tool-use checking.

    ``severity`` controls the response on match:
      - ``"advise"`` (default): emit ``additionalContext`` only. The tool
        proceeds; the agent gets a system reminder and decides for itself.
      - ``"block"``: opt-in hard stop. Emit ``permissionDecision: deny``; the
        tool is refused. Reserve for genuinely dangerous actions.

    Carries everything the decision formatter needs without re-reading the
    file: the absolute entry path (for the wikilink), the title, the one-line
    rule, and the severity.
    """

    entry_path: Path
    title: str
    one_line_rule: str
    severity: str = "advise"
    triggers: tuple[Trigger, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Decision:
    """The outcome of a blocker match, ready for the hook to emit.

    ``severity`` is ``"block"`` or ``"advise"``. ``hook_output`` is the dict
    the caller merges into ``{"hookSpecificOutput": {"hookEventName":
    "PreToolUse", **hook_output}}`` — a ``permissionDecision: deny`` payload
    for blocks, an ``additionalContext`` payload for advisories. ``wiki`` is
    the wikilink target of the matched entry, exposed so the caller can tag the
    daemon event payload (legacy ``pre_tool_use.py`` set ``blocker_entry`` /
    ``summary`` from it).
    """

    severity: str
    hook_output: dict[str, str]
    wiki: str


# ── Frontmatter parser (minimal, dependency-free) ────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter_text, body_text)``. Empty frontmatter if the file
    doesn't start with a ``---`` fence.
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
    """Pull out top-level ``key: value`` scalar pairs.

    Only catches the trivial single-line form. Multi-line / block-scalar / list
    values are ignored — they're either handled by ``_parse_block_triggers``
    (for triggers) or not relevant to this module's job.
    """
    out: dict[str, str] = {}
    for line in fm.splitlines():
        # Top-level keys have no leading indent. Skip indented lines (those
        # belong to a multi-line value) and skip list-item lines.
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

    Recognised shape (the only shape AGENTS.md documents)::

        block_triggers:
          - tool: Bash
            pattern: '<regex>'
          - tool: Edit
            file_pattern: '<regex>'

    Each list item starts with ``  - key: value`` and subsequent
    ``    key: value`` lines belong to the same item until the next ``-`` or
    until the block ends. Items that fail to parse cleanly are skipped rather
    than raised.
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

    Strips a leading ``#`` heading (the title is already in frontmatter) and
    caps at 200 chars — the deny reason becomes part of the agent's prompt, no
    need to embed a paragraph.
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


# ── Cache machinery ──────────────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_CACHE: dict[Path, tuple[float, list[Blocker]]] = {}


def _cache_sentinel(kdir: Path) -> float:
    """Return an mtime that ticks whenever the library changes.

    Prefers ``log.md`` (the Scholar appends to it on every write); falls back
    to the dir's own mtime; finally 0.0 if both are missing.
    """
    log_path = kdir / "log.md"
    try:
        return log_path.stat().st_mtime
    except OSError:
        pass
    try:
        return kdir.stat().st_mtime
    except OSError:
        return 0.0


def _safe_compile(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    """Compile ``pattern`` or return None on missing/invalid regex.

    Invalid regexes are silently dropped — graceful degradation beats crashing
    the host agent on every tool call.
    """
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _trigger_from_raw(raw: dict[str, str]) -> Optional[Trigger]:
    """Turn one parsed ``block_triggers`` entry into a :class:`Trigger`.

    Returns None if there's no tool name or if both regexes are empty/invalid
    (a trigger with neither is functionally inert).
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

    Opt-in is driven by the PRESENCE of ``block_triggers``. Returns ``None``
    for entries with no triggers (the common case), that fail to parse, or that
    have zero usable triggers (a blocker with no triggers is inert).
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


def _scan_blockers(kdir: Path) -> list[Blocker]:
    """Walk the knowledge tree once, collecting every blocker entry.

    Skips ``_archive/`` (archived entries shouldn't fire interrupts) and the
    top-level ``index.md`` / ``log.md`` / ``README.md`` admin files.
    """
    out: list[Blocker] = []
    if not kdir.exists():
        return out
    for path in kdir.rglob("*.md"):
        try:
            rel = path.relative_to(kdir)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] == "_archive":
            continue
        if rel.name in ("index.md", "log.md", "README.md"):
            continue
        blocker = _build_blocker(path)
        if blocker is not None:
            out.append(blocker)
    return out


# ── Loader (cached) ──────────────────────────────────────────────────────


def load_blockers(kdir: Path) -> list[Blocker]:
    """Return every entry under ``kdir`` that opts in to blocking.

    Cached per ``(kdir, sentinel_mtime)``. The sentinel advances whenever
    ``log.md`` is touched; if it's missing we fall back to the directory's own
    mtime. Each call pays one ``stat()`` on the sentinel and a full rescan only
    if the mtime moved. Thread-safe under a single module-level lock.
    """
    key = kdir.resolve()
    sentinel = _cache_sentinel(key)

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == sentinel:
            return cached[1]
        blockers = _scan_blockers(key)
        _CACHE[key] = (sentinel, blockers)
        return blockers


def clear_cache() -> None:
    """Drop all cached blockers. Test-only — production hooks run in fresh
    subprocesses so the cache is naturally cold per call.
    """
    with _CACHE_LOCK:
        _CACHE.clear()


# ── Matcher ──────────────────────────────────────────────────────────────


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

    - A trigger fires only when ``trigger.tool`` equals ``tool_name``
      (case-sensitive; Claude Code's tool names are PascalCase: ``Bash``,
      ``Edit``, ``Write``, ``NotebookEdit``).
    - Bash triggers (``pattern``) match against ``command`` via ``re.search``.
    - Edit/Write/NotebookEdit triggers (``file_pattern``) match against
      ``file_path``.

    Returns the first match in iteration order so rule authors get stable
    behaviour when two blockers might both fire.
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


def rel_to_knowledge(entry_path: Path, kdir: Optional[Path] = None) -> str:
    """Format the entry path as a wikilink target.

    If ``kdir`` is provided and the entry sits under it, return the path
    relative to ``knowledge/`` without the ``.md`` extension. Otherwise
    best-effort slice from a ``knowledge`` path segment, falling back to the
    bare stem so the message still names something.
    """
    if kdir is not None:
        try:
            rel = entry_path.resolve().relative_to(kdir.resolve())
            return str(rel).replace(os.sep, "/").removesuffix(".md")
        except ValueError:
            pass
    parts = entry_path.parts
    for i, part in enumerate(parts):
        if part == "knowledge" and i + 1 < len(parts):
            tail = "/".join(parts[i + 1 :])
            return tail.removesuffix(".md")
    return entry_path.stem


# ── Decision formatting (ported from src/hooks/pre_tool_use.py) ───────────


def _block_output(wiki: str, rule: str) -> dict[str, str]:
    """The ``permissionDecision: deny`` payload for a hard-stop blocker."""
    reason = (
        f"⚠ Library blocks this action: [[{wiki}]] - {rule} Confirm with the user before retrying."
    )
    return {"permissionDecision": "deny", "permissionDecisionReason": reason}


def _advise_output(wiki: str, rule: str) -> dict[str, str]:
    """The ``additionalContext`` FYI payload for an advisory blocker."""
    notice = f"📚 Library note (FYI; agent decides): [[{wiki}]] applies here — {rule}"
    return {"additionalContext": notice}


# ── Public entry point ───────────────────────────────────────────────────


def evaluate(
    tool_name: str,
    tool_input: dict[str, Any],
    kdir: Optional[Path] = None,
) -> Optional[Decision]:
    """Evaluate a pending tool call against the library's blocker rules.

    Returns ``None`` when nothing matches (the common case — the caller lets
    the tool proceed and emits no ``hookSpecificOutput`` beyond its own event
    capture). On a match returns a :class:`Decision` carrying the severity, the
    ready-to-emit ``hook_output`` dict, and the matched entry's wikilink.

    Short-circuits to ``None`` when ``CLAUDE_INVOKED_BY`` is set: inside a
    daemon-spawned Agent-SDK subprocess (Scholar/Librarian) we must never block
    the agent's own Edit/Write calls or we deadlock the daemon. Keeping the
    guard here means the Phase-2 dispatch branch can't forget it.

    Never raises — any failure resolves to ``None`` (no block), matching the
    legacy hook's "the blocker check must never crash the host agent" contract.
    The worst case is "no block", identical to running without this tier.

    Args:
        tool_name: Claude Code tool name (PascalCase: ``Bash``, ``Edit``, …).
        tool_input: the ``tool_input`` dict from the hook stdin payload.
        kdir: knowledge dir override (tests pass a tmp dir). Defaults to
            :func:`knowledge_dir` — ``${AGENT_MEM_HOME:-~/.agent-mem}/knowledge``.

    To emit the decision, merge ``decision.hook_output`` into the PreToolUse
    envelope::

        {"hookSpecificOutput": {"hookEventName": "PreToolUse", **decision.hook_output}}
    """
    # Recursion guard — must come before any work. See docstring for why this
    # is non-negotiable for daemon stability.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return None

    resolved = kdir if kdir is not None else knowledge_dir()
    try:
        blockers = load_blockers(resolved)
        match = find_match(blockers, tool_name, tool_input)
    except Exception:
        # The blocker check must never crash the host agent — worst case is
        # "no block", same as today's behaviour without this tier.
        return None

    if match is None:
        return None

    rule = match.one_line_rule or "(no inline rule text)"
    try:
        wiki = rel_to_knowledge(match.entry_path, resolved)
    except Exception:
        wiki = match.entry_path.stem

    if match.severity == "block":
        return Decision(severity="block", hook_output=_block_output(wiki, rule), wiki=wiki)
    return Decision(severity="advise", hook_output=_advise_output(wiki, rule), wiki=wiki)
