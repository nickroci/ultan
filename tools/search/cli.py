"""`agent-mem` CLI — search + lifecycle (review/promote/demote/forget/doctor).

Search modes (per PLAN section 3):

  --hierarchy <subpath>
      Walk `~/.agent-mem/knowledge/<subpath>` and print every `.md` path found.
      No LLM, no scoring. Just `ls -R`.

  --index <query>
      Index-led LLM retrieval. Loads `~/.agent-mem/knowledge/index.md` and asks
      Claude Agent SDK (with allowed_tools=["Read"]) which entries are
      relevant, then to read them and answer.

  --bm25 <query>
      Stock BM25 over article bodies. Uses a persisted index at
      `~/.agent-mem/.bm25.idx`, rebuilt on demand when sources change.

  (no flag) <query>
      Default behavior. Runs BM25 + index-led retrieval (hierarchy is a
      browse tool, not a query tool) and merges results, deduplicated by
      file path. One hit per file, with a one-line snippet and the mode(s)
      it surfaced from.

Lifecycle subcommands (per PLAN section 6):

  review                      Interactive batch promote/reject of provisional entries.
  promote <id-or-path>        Set status: confirmed.
  demote  <id-or-path>        Set status: provisional.
  forget  <id-or-path>        Move to _archive/ (relative path preserved).
  doctor                      Lint + daemon liveness + cost + corpus stats.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable, TextIO

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from bm25 import load_or_build
from frontmatter import (
    FrontmatterError,
    set_status,
)
from frontmatter import (
    read as fm_read,
)

DEFAULT_AGENT_MEM_ROOT = Path("~/.agent-mem").expanduser()
DEFAULT_KNOWLEDGE_DIR = DEFAULT_AGENT_MEM_ROOT / "knowledge"


# ── Knowledge dir resolution ───────────────────────────────────────────────────


def _resolve_knowledge_dir(explicit: str | None) -> Path:
    """Resolve knowledge dir from --knowledge-dir, $AGENT_MEM_KNOWLEDGE, or default."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("AGENT_MEM_KNOWLEDGE")
    if env:
        return Path(env).expanduser().resolve()
    home = os.environ.get("AGENT_MEM_HOME")
    if home:
        return (Path(home).expanduser() / "knowledge").resolve()
    return DEFAULT_KNOWLEDGE_DIR


def _resolve_agent_mem_home(knowledge_dir: Path) -> Path:
    """Where things like cost.json and daemon.pid live.

    Prefers ``$AGENT_MEM_HOME``; otherwise infers from the knowledge dir
    (parent), which matches the layout in `daemon/agent_mem_daemon/paths.py`.
    """
    env = os.environ.get("AGENT_MEM_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return knowledge_dir.parent


# ── Hierarchy mode ─────────────────────────────────────────────────────────────


def hierarchy_mode(knowledge_dir: Path, subpath: str | None) -> int:
    """List every .md under knowledge_dir/subpath (or the whole tree)."""
    if not knowledge_dir.exists():
        print(f"knowledge dir not found: {knowledge_dir}", file=sys.stderr)
        return 2

    root = knowledge_dir / subpath if subpath else knowledge_dir
    root = root.resolve()
    # Refuse to escape the knowledge dir.
    try:
        root.relative_to(knowledge_dir)
    except ValueError:
        print(f"refusing to walk outside knowledge dir: {root}", file=sys.stderr)
        return 2

    if not root.exists():
        print(f"path not found: {root}", file=sys.stderr)
        return 1

    if root.is_file():
        print(root)
        return 0

    found = False
    for p in sorted(root.rglob("*.md")):
        if "_archive" in p.parts:
            continue
        print(p)
        found = True
    if not found:
        print(f"(no markdown entries under {root})")
    return 0


# ── BM25 mode ──────────────────────────────────────────────────────────────────


@dataclass
class Hit:
    path: Path
    score: float
    snippet: str
    sources: list[str]


def bm25_mode(
    knowledge_dir: Path,
    query: str,
    k: int = 10,
    force_rebuild: bool = False,
) -> list[Hit]:
    """Return BM25 hits (also used by merged mode)."""
    if not knowledge_dir.exists():
        return []
    index = load_or_build(knowledge_dir, force_rebuild=force_rebuild)
    raw = index.search(query, k=k)
    return [Hit(path=p, score=score, snippet=snip, sources=["bm25"]) for p, score, snip in raw]


# ── Index-led LLM mode ─────────────────────────────────────────────────────────


_INDEX_PROMPT_TEMPLATE = """You are a knowledge base query engine for `agent-mem`.

The user's question is below. The knowledge base lives at:

    {knowledge_dir}

The master catalog is at:

    {index_file}

Procedure:

1. Read `{index_file}` first. It lists every entry with a short summary and
   (where present) the `applies-when` triggers.
2. Pick 1-10 entries that look relevant. If none look relevant, say so
   explicitly — do not fabricate.
3. Use the Read tool to open each candidate in full.
4. Synthesize a clear answer. Cite every source as an absolute path on its
   own line, prefixed with `SOURCE: `. The CLI parses those lines.

Question:

{question}
"""


async def _drain_sdk_response(
    options_factory: Callable[[], ClaudeAgentOptions],
    prompt: str,
) -> tuple[str, str | None]:
    """Stream assistant text from the SDK; return (joined_text, error_or_None)."""
    chunks: list[str] = []
    try:
        async for message in query(prompt=prompt, options=options_factory()):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
    except Exception as e:  # noqa: BLE001 — surface SDK failure to user
        return "", f"[index mode error: {e}]"
    return "".join(chunks), None


def _collect_citations(answer: str) -> list[Path]:
    """Pull SOURCE: lines out of the assistant answer, deduped, existing-only."""
    cited: list[Path] = []
    seen: set[str] = set()
    for line in answer.splitlines():
        if not line.startswith("SOURCE:"):
            continue
        raw = line[len("SOURCE:") :].strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        resolved = p.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            cited.append(resolved)
    return cited


async def _run_index_query(knowledge_dir: Path, question: str) -> tuple[str, list[Path]]:
    """Call Claude Agent SDK to answer via index-led retrieval.

    Returns (answer_text, list_of_cited_paths).
    """
    index_file = knowledge_dir / "index.md"
    if not index_file.exists():
        return (
            f"[index mode: {index_file} does not exist yet — nothing to search]",
            [],
        )

    prompt = _INDEX_PROMPT_TEMPLATE.format(
        knowledge_dir=knowledge_dir,
        index_file=index_file,
        question=question,
    )

    def _make_options():
        return ClaudeAgentOptions(
            cwd=str(knowledge_dir),
            system_prompt={"type": "preset", "preset": "claude_code"},
            allowed_tools=["Read"],
            permission_mode="acceptEdits",
            max_turns=15,
        )

    answer, err = await _drain_sdk_response(_make_options, prompt)
    if err is not None:
        return err, []

    return answer, _collect_citations(answer)


def index_mode(knowledge_dir: Path, question: str) -> tuple[str, list[Hit]]:
    """Sync wrapper. Returns (LLM answer, hits derived from cited sources)."""
    answer, cited = asyncio.run(_run_index_query(knowledge_dir, question))
    hits = [
        Hit(path=p, score=1.0, snippet="(cited by index-led retrieval)", sources=["index"])
        for p in cited
    ]
    return answer, hits


# ── Merged default mode ────────────────────────────────────────────────────────


def merged_mode(knowledge_dir: Path, query: str, k: int = 10) -> tuple[list[Hit], str]:
    """Run BM25 + index-led, merge by file path. Returns (hits, llm_answer)."""
    bm25_hits = bm25_mode(knowledge_dir, query, k=k)
    llm_answer, index_hits = index_mode(knowledge_dir, query)

    by_path: dict[str, Hit] = {}
    for h in bm25_hits + index_hits:
        key = str(h.path.resolve())
        if key not in by_path:
            by_path[key] = Hit(
                path=h.path,
                score=h.score,
                snippet=h.snippet,
                sources=list(h.sources),
            )
        else:
            existing = by_path[key]
            # Prefer a BM25 snippet over the generic index snippet.
            if "bm25" in h.sources and "bm25" not in existing.sources:
                existing.snippet = h.snippet
            for src in h.sources:
                if src not in existing.sources:
                    existing.sources.append(src)
            existing.score = max(existing.score, h.score)

    merged = sorted(
        by_path.values(),
        key=lambda h: (-len(h.sources), -h.score),
    )
    return merged, llm_answer


# ── Printing ───────────────────────────────────────────────────────────────────


def _print_hits(hits: list[Hit], header: str | None = None) -> None:
    if header:
        print(header)
    if not hits:
        print("  (no hits)")
        return
    for h in hits:
        srcs = "+".join(h.sources)
        print(f"  [{srcs:>10}]  {h.score:6.2f}  {h.path}")
        if h.snippet:
            print(f"               {h.snippet}")


# ── Lifecycle helpers ──────────────────────────────────────────────────────────


def _iter_entries(knowledge_dir: Path) -> Iterable[Path]:
    """Every live `.md` entry under knowledge_dir.

    Excludes `_archive/` and the catalog files (`index.md`, `log.md`).
    Mirrors what BM25 indexes so the two views agree.
    """
    catalog_names = {"index.md", "log.md"}
    for p in sorted(knowledge_dir.rglob("*.md")):
        if "_archive" in p.parts:
            continue
        if p.parent == knowledge_dir and p.name in catalog_names:
            continue
        yield p


def _safe_read_frontmatter(path: Path) -> dict | None:
    """Return frontmatter dict or None on error. For tolerant scans."""
    try:
        fm, _ = fm_read(path)
    except (FrontmatterError, OSError, UnicodeDecodeError):
        return None
    return fm if fm else None


def _resolve_identifier(knowledge_dir: Path, ident: str) -> Path:
    """Resolve a CLI identifier (an `id` value or a path) to a single Path.

    Path resolution accepts: absolute paths, paths relative to the knowledge
    root, with or without ``.md``. Falls back to scanning frontmatter ``id:``
    values across the corpus; if there are multiple matches we raise
    ``LookupError`` listing them so the caller can show the user.
    """
    # Try as a path first.
    for candidate in _identifier_path_candidates(knowledge_dir, ident):
        if candidate.is_file():
            return candidate.resolve()

    # Otherwise scan for an id: <ident> frontmatter match.
    matches: list[Path] = []
    for p in _iter_entries(knowledge_dir):
        fm = _safe_read_frontmatter(p)
        if fm and str(fm.get("id", "")).strip() == ident:
            matches.append(p)
    if not matches:
        raise LookupError(f"no entry found for {ident!r} (tried path and id: lookup)")
    if len(matches) > 1:
        listing = "\n  ".join(str(p) for p in matches)
        raise LookupError(
            f"identifier {ident!r} is ambiguous; matches:\n  {listing}\n"
            "Pass a specific path instead."
        )
    return matches[0].resolve()


def _identifier_path_candidates(knowledge_dir: Path, ident: str) -> list[Path]:
    """Generate plausible paths for an id-or-path argument."""
    candidates: list[Path] = []
    raw = Path(ident).expanduser()
    if raw.is_absolute():
        candidates.append(raw)
        if raw.suffix != ".md":
            candidates.append(raw.with_suffix(".md"))
    else:
        # Relative to cwd.
        candidates.append(raw)
        if raw.suffix != ".md":
            candidates.append(raw.with_suffix(".md"))
        # Relative to knowledge_dir.
        candidates.append(knowledge_dir / raw)
        if raw.suffix != ".md":
            candidates.append((knowledge_dir / raw).with_suffix(".md"))
    return candidates


def _short_id(knowledge_dir: Path, path: Path) -> str:
    """Render a path nicely for messages — id if available, else relative path."""
    fm = _safe_read_frontmatter(path)
    if fm and fm.get("id"):
        return str(fm["id"])
    try:
        return str(path.relative_to(knowledge_dir))
    except ValueError:
        return str(path)


# ── promote / demote / forget ──────────────────────────────────────────────────


def cmd_promote(knowledge_dir: Path, ident: str) -> int:
    try:
        path = _resolve_identifier(knowledge_dir, ident)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    fm, _ = fm_read(path)
    current = str(fm.get("status", "")).strip()
    if current == "confirmed":
        print(f"{_short_id(knowledge_dir, path)} is already confirmed (no-op)", file=sys.stderr)
        return 1
    set_status(path, "confirmed")
    print(f"promoted {_short_id(knowledge_dir, path)} -> confirmed  ({path})")
    return 0


def cmd_demote(knowledge_dir: Path, ident: str) -> int:
    try:
        path = _resolve_identifier(knowledge_dir, ident)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    fm, _ = fm_read(path)
    current = str(fm.get("status", "")).strip()
    if current == "provisional":
        print(
            f"{_short_id(knowledge_dir, path)} is already provisional (no-op)",
            file=sys.stderr,
        )
        return 1
    set_status(path, "provisional")
    print(f"demoted {_short_id(knowledge_dir, path)} -> provisional  ({path})")
    return 0


def cmd_forget(knowledge_dir: Path, ident: str) -> int:
    try:
        path = _resolve_identifier(knowledge_dir, ident)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        rel = path.relative_to(knowledge_dir)
    except ValueError:
        print(
            f"refusing to archive {path}: not under knowledge dir {knowledge_dir}",
            file=sys.stderr,
        )
        return 2
    archive_dir = knowledge_dir / "_archive"
    target = archive_dir / rel
    if target.exists():
        print(f"archive target already exists: {target}", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    print(f"archived {_short_id(knowledge_dir, target)}  ({path} -> {target})")
    return 0


# ── review ────────────────────────────────────────────────────────────────────


def _list_provisional(knowledge_dir: Path) -> list[Path]:
    """All provisional entries under knowledge_dir, in a stable order."""
    out: list[Path] = []
    for p in _iter_entries(knowledge_dir):
        fm = _safe_read_frontmatter(p)
        if fm and str(fm.get("status", "")).strip() == "provisional":
            out.append(p)
    return out


def _entry_preview(path: Path, body_lines: int = 30) -> str:
    """A short, copy-pasteable preview for review."""
    fm = _safe_read_frontmatter(path) or {}
    body = ""
    try:
        _, body = fm_read(path)
    except FrontmatterError:
        pass
    head = [
        f"id        : {fm.get('id', '(missing)')}",
        f"path      : {path}",
        f"scope     : {fm.get('scope', '(missing)')}",
        f"status    : {fm.get('status', '(missing)')}",
        f"confidence: {fm.get('confidence', '(missing)')}",
    ]
    aw = fm.get("applies-when") or fm.get("applies_when")
    if isinstance(aw, str):
        for line in aw.splitlines():
            line = line.strip()
            if line:
                head.append(f"applies-when: {line}")
    elif isinstance(aw, list):
        for item in aw:
            head.append(f"applies-when: {item}")
    head.append("")
    snippet_lines = [ln for ln in body.splitlines()][:body_lines]
    return "\n".join(head + snippet_lines)


def _edit_in_editor(path: Path) -> None:
    """Spawn $EDITOR (fallback vi) on path. Blocks until exit."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    subprocess.call([editor, str(path)])


def _print_noninteractive_review(knowledge_dir: Path, entries: list[Path]) -> None:
    total = len(entries)
    print(f"{total} provisional entr{'y' if total == 1 else 'ies'} would be reviewed:")
    for i, p in enumerate(entries, start=1):
        short = _short_id(knowledge_dir, p)
        print(f"  [{i}/{total}] {short}  ({p})")


class _ReviewSignal(StrEnum):
    NEXT = "next"  # done with this entry, move on
    QUIT = "quit"  # exit cmd_review immediately
    UNKNOWN = "unknown"  # re-prompt for action


def _apply_review_action(action: str, path: Path, knowledge_dir: Path) -> _ReviewSignal:
    """Apply a single review action to ``path``. Returns whether to move on, quit, or re-prompt."""
    if action in {"p", "promote"}:
        try:
            set_status(path, "confirmed")
            print("  -> promoted to confirmed")
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed to promote: {e}", file=sys.stderr)
        return _ReviewSignal.NEXT
    if action in {"r", "reject"}:
        rc = cmd_forget(knowledge_dir, str(path))
        if rc != 0:
            print("  ! archive failed (see above)", file=sys.stderr)
        return _ReviewSignal.NEXT
    if action in {"e", "edit"}:
        _edit_in_editor(path)
        # Re-read to surface user changes before moving on.
        try:
            fm_after, _ = fm_read(path)
            print(f"  -> edited (status now: {fm_after.get('status', '?')})")
        except FrontmatterError as e:
            print(f"  ! file no longer parses: {e}", file=sys.stderr)
        return _ReviewSignal.NEXT
    if action in {"s", "skip", ""}:
        print("  -> skipped")
        return _ReviewSignal.NEXT
    if action in {"q", "quit"}:
        print("Quitting; remaining entries untouched.")
        return _ReviewSignal.QUIT
    print(f"  (unknown action: {action!r})")
    return _ReviewSignal.UNKNOWN


def cmd_review(
    knowledge_dir: Path,
    *,
    noninteractive: bool = False,
    stdin: "TextIO | None" = None,
) -> int:
    """Walk every provisional entry; prompt for [p]romote / [r]eject / [e]dit / [s]kip / [q]uit.

    When ``noninteractive`` is true, prints what *would* be reviewed and
    exits 0 without prompting. Used in CI.
    """
    entries = _list_provisional(knowledge_dir)
    total = len(entries)
    if total == 0:
        print("No provisional entries to review.")
        return 0

    if noninteractive:
        _print_noninteractive_review(knowledge_dir, entries)
        return 0

    in_stream = stdin if stdin is not None else sys.stdin

    print(f"Reviewing {total} provisional entr{'y' if total == 1 else 'ies'}.")
    print("Actions: [p]romote  [r]eject (archive)  [e]dit  [s]kip  [q]uit\n")

    for i, p in enumerate(entries, start=1):
        print("=" * 70)
        print(f"[{i}/{total}]")
        print(_entry_preview(p))
        print("=" * 70)

        while True:
            print("Action [p/r/e/s/q]: ", end="", flush=True)
            raw = in_stream.readline()
            if not raw:
                print("\n(no more input — quitting)")
                return 0
            signal = _apply_review_action(raw.strip().lower(), p, knowledge_dir)
            if signal is _ReviewSignal.QUIT:
                return 0
            if signal is _ReviewSignal.NEXT:
                break
            # UNKNOWN — loop and re-prompt
    return 0


# ── doctor ────────────────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — treat as alive.
        return True
    except OSError:
        return False
    return True


def _read_pid_file(pid_path: Path) -> int | None:
    try:
        text = pid_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _load_cost_state(cost_path: Path) -> dict | None:
    try:
        return json.loads(cost_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _scope_label(fm: dict) -> str:
    scope = str(fm.get("scope") or "(unset)")
    if scope.startswith("project:"):
        return scope
    return scope


def _run_structural_lint(knowledge_dir: Path) -> tuple[int, str]:
    """Best-effort: invoke src/scripts/lint.py via uv if reachable.

    Returns (exit_code, captured_output). exit_code < 0 means "could not run".
    """
    # Resolve src dir: same layout as the agent-mem checkout.
    src_dir = _find_src_dir()
    if not src_dir:
        return -1, "(lint skipped: could not locate src/ checkout)"
    script = src_dir / "scripts" / "lint.py"
    if not script.exists():
        return -1, f"(lint skipped: {script} not found)"
    env = os.environ.copy()
    # Point lint at the same knowledge dir we're operating on.
    env["AGENT_MEM_HOME"] = str(knowledge_dir.parent)
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "--directory",
                str(src_dir),
                "python",
                "scripts/lint.py",
                "--structural-only",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        return -1, "(lint skipped: `uv` not on PATH)"
    except subprocess.TimeoutExpired:
        return -1, "(lint skipped: timed out after 120s)"
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def _find_src_dir() -> Path | None:
    """Look for the agent-mem `src/` checkout next to this CLI.

    The CLI lives in `<repo>/tools/search/cli.py`; src is `<repo>/src/`.
    """
    here = Path(__file__).resolve().parent
    candidate = here.parent.parent / "src"
    if (candidate / "scripts" / "lint.py").exists():
        return candidate
    # Fall back to AGENT_MEM_SRC if set.
    env = os.environ.get("AGENT_MEM_SRC")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "scripts" / "lint.py").exists():
            return p
    return None


@dataclass
class DoctorReport:
    """Structured doctor results — useful for tests and for the CLI exit code."""

    lint_rc: int
    lint_output: str
    daemon_status: str  # "running" | "stale" | "absent"
    daemon_pid: int | None
    cost_today: float
    cost_lifetime: float
    counts_by_status: dict[str, int]
    counts_by_scope: dict[str, int]
    total_entries: int
    pending_nudges: int  # -1 if file absent
    issues: list[str]  # human-readable strings; non-empty ⇒ rc != 0


def run_doctor(knowledge_dir: Path) -> DoctorReport:
    """Gather everything `agent-mem doctor` reports, without printing."""
    home = _resolve_agent_mem_home(knowledge_dir)

    # Lint (structural only — cheap; the LLM contradictions check costs money).
    lint_rc, lint_output = _run_structural_lint(knowledge_dir)

    # Daemon liveness.
    pid_path = home / "daemon.pid"
    pid = _read_pid_file(pid_path)
    if pid is None:
        daemon_status = "absent"
    elif _pid_alive(pid):
        daemon_status = "running"
    else:
        daemon_status = "stale"

    # Cost.
    cost_path = home / "cost.json"
    cost_state = _load_cost_state(cost_path) or {}
    cost_today = float(cost_state.get("today_usd") or 0.0)
    cost_lifetime = float(cost_state.get("lifetime_usd") or 0.0)
    # If "today" rolled over the daemon would reset; doctor doesn't write.
    today = date.today().isoformat()
    if str(cost_state.get("today")) != today:
        cost_today = 0.0

    # Corpus stats.
    counts_by_status: dict[str, int] = {}
    counts_by_scope: dict[str, int] = {}
    total = 0
    if knowledge_dir.exists():
        for p in _iter_entries(knowledge_dir):
            fm = _safe_read_frontmatter(p) or {}
            status = str(fm.get("status") or "(unset)")
            counts_by_status[status] = counts_by_status.get(status, 0) + 1
            scope = _scope_label(fm)
            counts_by_scope[scope] = counts_by_scope.get(scope, 0) + 1
            total += 1

    # Pending nudges.
    nudges_path = home / "pending-nudges.md"
    if nudges_path.exists():
        try:
            text = nudges_path.read_text(encoding="utf-8")
            # Treat each non-blank line that isn't a header as a nudge.
            pending_nudges = sum(
                1
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        except OSError:
            pending_nudges = -1
    else:
        pending_nudges = -1

    issues: list[str] = []
    if lint_rc > 0:
        issues.append(f"lint reported issues (exit {lint_rc})")
    # Daemon status is informational only — we don't enforce a "should be running"
    # rule here (the user might run agent-mem without ever starting the daemon).
    # Stale PIDs are worth flagging since they indicate a crash.
    if daemon_status == "stale":
        issues.append(f"daemon PID {pid} in {pid_path} is not alive — stale pidfile")

    return DoctorReport(
        lint_rc=lint_rc,
        lint_output=lint_output,
        daemon_status=daemon_status,
        daemon_pid=pid,
        cost_today=cost_today,
        cost_lifetime=cost_lifetime,
        counts_by_status=counts_by_status,
        counts_by_scope=counts_by_scope,
        total_entries=total,
        pending_nudges=pending_nudges,
        issues=issues,
    )


def _print_doctor_header(home: Path, knowledge_dir: Path) -> None:
    print("agent-mem doctor")
    print("=" * 60)
    print(f"home          : {home}")
    print(f"knowledge dir : {knowledge_dir}")
    print()


def _print_doctor_corpus(report: DoctorReport) -> None:
    print(f"Corpus: {report.total_entries} entr{'y' if report.total_entries == 1 else 'ies'}")
    if report.counts_by_status:
        print("  by status:")
        for status, count in sorted(report.counts_by_status.items()):
            print(f"    {status:>15}  {count}")
    if report.counts_by_scope:
        print("  by scope:")
        for scope, count in sorted(report.counts_by_scope.items()):
            print(f"    {scope:>25}  {count}")
    print()


_DAEMON_MARKERS = {"running": "[OK]", "stale": "[!!]"}


def _print_doctor_daemon(report: DoctorReport) -> None:
    marker = _DAEMON_MARKERS.get(report.daemon_status, "[--]")
    print(f"Daemon: {marker} {report.daemon_status}", end="")
    if report.daemon_pid is not None:
        print(f"  (pid {report.daemon_pid})")
    else:
        print()
    print()


def _print_doctor_cost(report: DoctorReport) -> None:
    print(
        f"Cost: today ${report.cost_today:.4f}; lifetime ${report.cost_lifetime:.4f}  "
        f"(cost cap: disabled)"
    )
    print()


def _print_doctor_nudges(report: DoctorReport) -> None:
    if report.pending_nudges < 0:
        print("Pending nudges: (no pending-nudges.md)")
    else:
        print(f"Pending nudges: {report.pending_nudges}")
    print()


def _print_doctor_lint(report: DoctorReport) -> None:
    print("Lint (structural-only):")
    print("-" * 60)
    if report.lint_rc < 0:
        print(report.lint_output.strip() or "(lint unavailable)")
    else:
        print(report.lint_output.strip() or "(no output)")
        print(f"(exit {report.lint_rc})")
    print("-" * 60)


def cmd_doctor(knowledge_dir: Path) -> int:
    report = run_doctor(knowledge_dir)
    home = _resolve_agent_mem_home(knowledge_dir)

    _print_doctor_header(home, knowledge_dir)
    _print_doctor_corpus(report)
    _print_doctor_daemon(report)
    _print_doctor_cost(report)
    _print_doctor_nudges(report)
    _print_doctor_lint(report)

    if report.issues:
        print()
        print("Issues:")
        for issue in report.issues:
            print(f"  - {issue}")
        return 1
    print()
    print("All checks clean.")
    return 0


# ── Argparse wiring ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-mem",
        description=(
            "agent-mem CLI. The `search` subcommand exposes three retrieval modes "
            "over `~/.agent-mem/knowledge/`:\n"
            "  --hierarchy <subpath> : list .md entries under a subdir (no LLM).\n"
            "  --index <query>       : index-led LLM retrieval via Claude Agent SDK.\n"
            "  --bm25 <query>        : keyword search over article bodies.\n"
            "  (no flag) <query>     : runs BM25 + index-led, merges by file path, "
            "one hit per file with the mode(s) that surfaced it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--knowledge-dir",
        help=(
            "Override the knowledge dir (default: $AGENT_MEM_KNOWLEDGE or ~/.agent-mem/knowledge)."
        ),
    )

    sub = p.add_subparsers(dest="command", required=True)

    search = sub.add_parser(
        "search",
        help="Search the knowledge store.",
        description=(
            "Three modes, plus a merged default. With no mode flag, runs BM25 and "
            "index-led retrieval in parallel and presents a deduplicated list of "
            "files (one hit per file), tagged with the mode(s) that surfaced them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = search.add_mutually_exclusive_group()
    group.add_argument(
        "--hierarchy",
        nargs="?",
        const="",
        metavar="SUBPATH",
        help="List .md entries under knowledge_dir/SUBPATH (no SUBPATH lists the whole tree).",
    )
    group.add_argument("--index", action="store_true", help="Index-led LLM retrieval.")
    group.add_argument("--bm25", action="store_true", help="BM25 keyword search.")
    search.add_argument(
        "query",
        nargs="*",
        help="Query string (omit for --hierarchy without subpath).",
    )
    search.add_argument("-k", type=int, default=10, help="Max BM25 hits to return (default 10).")
    search.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a BM25 index rebuild even if the cache looks fresh.",
    )

    # ── Lifecycle subcommands ─────────────────────────────────────────────────
    review = sub.add_parser(
        "review",
        help="Interactively review provisional entries.",
        description=(
            "Walk every entry with `status: provisional` and prompt for "
            "[p]romote / [r]eject / [e]dit / [s]kip / [q]uit."
        ),
    )
    review.add_argument(
        "--noninteractive",
        action="store_true",
        help="Print what would be reviewed and exit (useful in CI).",
    )

    promote = sub.add_parser(
        "promote",
        help="Set an entry's status to `confirmed`.",
    )
    promote.add_argument(
        "identifier",
        help="Entry id or path (absolute / relative to knowledge root).",
    )

    demote = sub.add_parser(
        "demote",
        help="Set an entry's status to `provisional`.",
    )
    demote.add_argument("identifier", help="Entry id or path.")

    forget = sub.add_parser(
        "forget",
        help="Move an entry to `_archive/`.",
    )
    forget.add_argument("identifier", help="Entry id or path.")

    sub.add_parser(
        "doctor",
        help="Run lint, check daemon liveness, summarise costs and corpus stats.",
    )

    return p


class Subcommand(StrEnum):
    """Subcommand names — must match the strings registered in ``_build_parser``."""

    SEARCH = "search"
    REVIEW = "review"
    PROMOTE = "promote"
    DEMOTE = "demote"
    FORGET = "forget"
    DOCTOR = "doctor"


def _run_search_bm25(knowledge_dir: Path, query_text: str, args: argparse.Namespace) -> int:
    hits = bm25_mode(knowledge_dir, query_text, k=args.k, force_rebuild=args.rebuild)
    _print_hits(hits, header=f"BM25 results for: {query_text!r}")
    if not hits:
        print("\n(no BM25 hits — try --index for paraphrase-tolerant retrieval)")
    return 0


def _run_search_index(knowledge_dir: Path, query_text: str) -> int:
    answer, hits = index_mode(knowledge_dir, query_text)
    print(f"Index-led answer for: {query_text!r}")
    print("-" * 60)
    print(answer.strip() or "(no answer)")
    print("-" * 60)
    _print_hits(hits, header="Entries cited:")
    return 0


def _run_search_merged(knowledge_dir: Path, query_text: str, k: int) -> int:
    print(f"Merged search for: {query_text!r}")
    print("Running BM25 + index-led retrieval, deduplicating by file path…")
    print()
    hits, llm_answer = merged_mode(knowledge_dir, query_text, k=k)
    _print_hits(hits, header="Entries:")

    print()
    print("Index-led synthesis:")
    print("-" * 60)
    print(llm_answer.strip() or "(no synthesis)")
    print("-" * 60)

    if not hits and (not llm_answer.strip() or llm_answer.startswith("[")):
        print()
        print(
            "Nothing matched. The knowledge store may be empty or your query "
            "doesn't intersect any stored entries."
        )
    return 0


def _dispatch_search(
    knowledge_dir: Path, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    # Hierarchy mode — no query needed.
    if args.hierarchy is not None:
        subpath = args.hierarchy or None
        return hierarchy_mode(knowledge_dir, subpath)

    query_text = " ".join(args.query).strip()
    if not query_text:
        parser.error("query is required unless --hierarchy is used")
        return 2  # unreachable; parser.error raises SystemExit

    if args.bm25:
        return _run_search_bm25(knowledge_dir, query_text, args)
    if args.index:
        return _run_search_index(knowledge_dir, query_text)
    return _run_search_merged(knowledge_dir, query_text, args.k)


# Dispatch table: each handler takes (knowledge_dir, args, parser) and returns an exit code.
# Centralising this means `main` is just "parse args, route to handler".
_HANDLERS: dict[Subcommand, Callable[[Path, argparse.Namespace, argparse.ArgumentParser], int]] = {
    Subcommand.SEARCH: _dispatch_search,
    Subcommand.REVIEW: lambda kd, args, _p: cmd_review(kd, noninteractive=args.noninteractive),
    Subcommand.PROMOTE: lambda kd, args, _p: cmd_promote(kd, args.identifier),
    Subcommand.DEMOTE: lambda kd, args, _p: cmd_demote(kd, args.identifier),
    Subcommand.FORGET: lambda kd, args, _p: cmd_forget(kd, args.identifier),
    Subcommand.DOCTOR: lambda kd, _args, _p: cmd_doctor(kd),
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    knowledge_dir = _resolve_knowledge_dir(args.knowledge_dir)

    try:
        sub = Subcommand(args.command)
    except ValueError:
        parser.error(f"unknown command: {args.command}")
        return 2  # unreachable; parser.error raises SystemExit

    return _HANDLERS[sub](knowledge_dir, args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
