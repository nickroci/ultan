"""In-process MCP tools the Librarian and Scholar can call.

This module exposes two SDK MCP tools, both running in-process so there
is no subprocess, no HTTP, no extra deps:

- ``bm25_search`` (Librarian + Scholar): wraps ``agent-mem-search`` BM25.
- ``move_entries`` (Scholar only): atomic multi-file move that creates
  the destination folder if missing, optionally writes its README, and
  rewrites every inbound wikilink in the library so the graph stays
  intact. Replaces the Scholar's old "Read + Write + Edit + remember to
  fix every back-reference" dance, which had been the dominant source
  of broken-wikilink violations.

The SDK exposes each tool to the model as
``mcp__<server_name>__<tool_name>``. ``allowed_tools`` entries in
llm.py whitelist the canonical names.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, cast

from bm25 import load_or_build
from claude_agent_sdk import create_sdk_mcp_server, tool
from embeddings import load_or_build as embeddings_load_or_build

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpServerConfig

log = logging.getLogger("agent_mem_daemon.library_tools")


# Server + tool naming. Keep stable; allowed_tools entries in llm.py
# refer to them.
SERVER_NAME = "agent_mem_library"
_BM25_TOOL_NAME = "bm25_search"
_EMBEDDING_TOOL_NAME = "embedding_search"
_MOVE_TOOL_NAME = "move_entries"


def fully_qualified_bm25_name() -> str:
    return f"mcp__{SERVER_NAME}__{_BM25_TOOL_NAME}"


def fully_qualified_embedding_name() -> str:
    return f"mcp__{SERVER_NAME}__{_EMBEDDING_TOOL_NAME}"


def fully_qualified_move_name() -> str:
    return f"mcp__{SERVER_NAME}__{_MOVE_TOOL_NAME}"


def fully_qualified_tool_name() -> str:
    """Backwards-compat shim — returns the bm25 name."""
    return fully_qualified_bm25_name()


def _text_response(text: str) -> Dict[str, Any]:
    """Wrap a plain string in the MCP content shape both search tools use."""
    return {"content": [{"type": "text", "text": text}]}


def unwrap_text_response(response: Dict[str, Any]) -> str:
    """Pull the plain text payload out of an MCP-shaped tool response
    (``{"content": [{"type": "text", "text": ...}]}``). Returns ``""`` when
    the shape is unexpected. Shared by the in-process callers (Scholar agent
    tools, executor) so the unwrap logic lives in one place."""
    content = response.get("content")
    if isinstance(content, list) and content:
        first = cast(object, content[0])
        if isinstance(first, dict):
            first_dict = cast(Dict[str, Any], first)
            return str(first_dict.get("text", ""))
    return ""


def _format_hit_lines(
    hits: Sequence[Tuple[Path, float, str]], root: Path, *, query: str
) -> Dict[str, Any]:
    """Render `(path, score, snippet)` tuples into the canonical text shape."""
    if not hits:
        return _text_response(f"(no results for {query!r})")
    lines = [
        f"{path.relative_to(root) if path.is_absolute() else path}  score={score:.2f}  {snippet}"
        for path, score, snippet in hits
    ]
    return _text_response("\n".join(lines))


def _parse_search_args(args: Dict[str, Any]) -> Tuple[str, int]:
    """Extract `query` and `k` from MCP-tool args (lenient on `k`)."""
    query = str(args.get("query") or "").strip()
    try:
        k = int(args.get("k") or 5)
    except (TypeError, ValueError):
        k = 5
    return query, max(1, min(20, k))


def run_bm25_search(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Public entry point for the BM25 search runner — same MCP-shaped
    response the in-process tool returns. Used directly (no subprocess) by
    the Scholar agent's ``bm25_search`` tool."""
    return _run_bm25_search(args, root)


def run_embedding_search(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Public entry point for the embedding search runner. Used directly by
    the Scholar agent's ``embedding_search`` tool."""
    return _run_embedding_search(args, root)


def move_entries(root: Path, args: Dict[str, Any]) -> Dict[str, Any]:
    """Public entry point for the atomic move + inbound-wikilink rewrite.
    Used directly by the Scholar executor (``move_entry`` action)."""
    return _move_entries_impl(root, args)


def _run_bm25_search(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
    query, k = _parse_search_args(args)
    if not query:
        return _text_response("(bm25_search: empty query)")
    if not root.exists():
        return _text_response("(library is empty — no entries to search)")
    try:
        index = load_or_build(root)
    except FileNotFoundError:
        return _text_response("(library has no entries yet)")
    except Exception as e:
        log.exception("bm25 index load/build failed")
        return _text_response(f"(bm25 backend error: {e})")
    return _format_hit_lines(index.search(query, k=k), root, query=query)


def _run_embedding_search(args: Dict[str, Any], root: Path) -> Dict[str, Any]:
    query, k = _parse_search_args(args)
    if not query:
        return _text_response("(embedding_search: empty query)")
    if not root.exists():
        return _text_response("(library is empty — no entries to search)")
    try:
        index = embeddings_load_or_build(root)
    except FileNotFoundError:
        return _text_response("(library has no entries yet)")
    except Exception as e:
        log.exception("embedding index load/build failed")
        return _text_response(f"(embedding backend error: {e})")
    raw = index.search(query, k=k)
    as_tuples: list[Tuple[Path, float, str]] = [(h.path, h.score, h.snippet) for h in raw]
    return _format_hit_lines(as_tuples, root, query=query)


def make_library_mcp_server(knowledge_dir: Path) -> McpServerConfig:
    """Build and return an SDK MCP server config exposing the library tools.

    Returns the value to pass into ``ClaudeAgentOptions.mcp_servers``
    under any dict key (the daemon uses ``SERVER_NAME``). The
    knowledge_dir is captured at construction time so the tools always
    search the same store — no path injection possible from the model.

    Exposed tools:
      - ``bm25_search`` — lexical relevance.
      - ``embedding_search`` — semantic similarity. Designed to be
        fanned out in parallel with ``bm25_search``.
      - ``move_entries`` — atomic multi-file move + wikilink rewrite.
    """
    root = knowledge_dir.expanduser().resolve()

    @tool(
        _BM25_TOOL_NAME,
        (
            "Search the agent-mem knowledge library by content relevance "
            "(BM25 ranking). Returns the top-K most relevant entries as "
            "lines of `<path>  score=<float>  <one-line snippet>`. Use "
            "this when you suspect an entry already covers a topic but "
            "don't see it in the library snapshot — it complements Glob "
            "(filename pattern) and Grep (literal regex). Run in parallel "
            "with `embedding_search` for any concept query. Typical k 3-8."
        ),
        {"query": str, "k": int},
    )
    async def bm25_search(args: Dict[str, Any]) -> Dict[str, Any]:
        return _run_bm25_search(args, root)

    @tool(
        _EMBEDDING_TOOL_NAME,
        (
            "Search the agent-mem knowledge library by semantic similarity "
            "(sentence-transformer embeddings, cosine similarity). Returns "
            "the top-K most similar entries as lines of `<path>  score=<float>  "
            "<one-line snippet>`. Complements `bm25_search` (lexical): call "
            "BOTH in parallel for any concept query — BM25 catches exact "
            "vocabulary matches, embeddings catch paraphrases. You may also "
            "fan out N parallel calls to either tool with different queries "
            "in a single turn (one tool_use block per query); they run "
            "concurrently. Typical k 3-8."
        ),
        {"query": str, "k": int},
    )
    async def embedding_search(args: Dict[str, Any]) -> Dict[str, Any]:
        return _run_embedding_search(args, root)

    @tool(
        _MOVE_TOOL_NAME,
        (
            "Atomically move one or more entries into a destination folder, "
            "rewriting every inbound wikilink in the library so the graph "
            "stays intact. Creates the destination folder if it does not "
            "exist, and (optionally) writes its README.md. Use this for "
            "BOTH single-file moves and split_folder operations — never "
            "move entries with Read+Write+Edit, wikilinks will break.\n\n"
            "Inputs (all paths relative to the knowledge root):\n"
            "  to_folder: destination folder (e.g. 'global/user/profile').\n"
            "  files: list of .md files to move into to_folder.\n"
            "  readme: optional content for to_folder/README.md. Skipped "
            "if a README already exists at that path.\n\n"
            "Returns: a summary of files moved, READMEs created, and "
            "wikilinks rewritten."
        ),
        {"to_folder": str, "files": list, "readme": str},
    )
    async def move_entries(args: Dict[str, Any]) -> Dict[str, Any]:
        return _move_entries_impl(root, args)

    return create_sdk_mcp_server(
        name=SERVER_NAME,
        version="1.0.0",
        tools=[bm25_search, embedding_search, move_entries],
    )


# ── move_entries implementation (pure-Python, no LLM in the loop) ─────


# Matches [[link]] or [[link|alias]]. Captures the link target (group 1)
# and the optional alias (group 2 incl. leading pipe, may be empty).
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\|]+?)(\|[^\[\]]*)?\]\]")


def _normalize_link(target: str) -> str:
    """Strip the optional .md suffix from a wikilink target."""
    t = target.strip()
    if t.endswith(".md"):
        t = t[:-3]
    return t


def _path_to_wikilink(path: Path, root: Path) -> str:
    """Convert an absolute file path to a wikilink target (no .md, posix)."""
    rel = path.resolve().relative_to(root)
    if rel.name.lower() == "readme.md":
        # Folder-shaped wikilinks: [[some/folder/]] resolves to its README.
        # Don't rewrite README files this way; callers shouldn't pass them
        # to move_entries anyway.
        return rel.as_posix()
    if rel.suffix == ".md":
        rel = rel.with_suffix("")
    return rel.as_posix()


def _safe_inside(root: Path, candidate: Path) -> bool:
    """True iff ``candidate`` resolves inside ``root`` (no path escape)."""
    try:
        candidate.resolve().relative_to(root)
    except (ValueError, OSError):
        return False
    return True


def _rewrite_wikilinks_in_text(text: str, mapping: Dict[str, str]) -> tuple[str, int]:
    """Rewrite every [[link]]/[[link|alias]] whose target matches a key
    in ``mapping`` to the corresponding value. Preserves the alias.
    Returns (new_text, rewrites_count)."""
    count = 0

    def repl(m: "re.Match[str]") -> str:
        nonlocal count
        raw_target = m.group(1)
        alias = m.group(2) or ""
        norm = _normalize_link(raw_target)
        new = mapping.get(norm)
        if new is None:
            return m.group(0)
        count += 1
        return f"[[{new}{alias}]]"

    return _WIKILINK_RE.sub(repl, text), count


def _move_err(msg: str) -> Dict[str, Any]:
    """Wire-shaped error response for move_entries."""
    return {"content": [{"type": "text", "text": f"(move_entries error) {msg}"}]}


def _resolve_destination(root: Path, raw_to: str) -> tuple[Path | None, Dict[str, Any] | None]:
    """Resolve and validate the destination folder. Returns
    ``(to_folder, None)`` on success or ``(None, err_response)``."""
    to_folder = (root / raw_to).resolve()
    if not _safe_inside(root, to_folder):
        return None, _move_err(f"to_folder {raw_to!r} resolves outside the knowledge root")
    return to_folder, None


def _plan_moves(
    root: Path,
    to_folder: Path,
    raw_files: List[object],
) -> tuple[list[tuple[Path, Path]] | None, Dict[str, str], Dict[str, Any] | None]:
    """Validate each source path and build the (src, dst) plan + the
    wikilink-rewrite mapping. Returns ``(moves, link_mapping, None)`` on
    success or ``(None, {}, err_response)`` on the first validation
    failure."""
    moves: List[tuple[Path, Path]] = []
    link_mapping: Dict[str, str] = {}
    for raw in raw_files:
        if not isinstance(raw, str) or not raw.strip():
            return (
                None,
                {},
                _move_err(f"each entry in 'files' must be a non-empty string; got {raw!r}"),
            )
        src = (root / raw).resolve()
        if not _safe_inside(root, src):
            return None, {}, _move_err(f"source {raw!r} resolves outside the knowledge root")
        if not src.exists():
            return None, {}, _move_err(f"source {raw!r} does not exist")
        if not src.is_file():
            return None, {}, _move_err(f"source {raw!r} is not a regular file")
        if src.suffix != ".md":
            return None, {}, _move_err(f"source {raw!r} is not a .md file")
        dst = (to_folder / src.name).resolve()
        if dst.exists() and dst != src:
            return (
                None,
                {},
                _move_err(
                    f"destination {dst.relative_to(root)} already exists; refusing to overwrite"
                ),
            )
        moves.append((src, dst))
        link_mapping[_path_to_wikilink(src, root)] = _path_to_wikilink(dst, root)
    return moves, link_mapping, None


def _maybe_write_readme(to_folder: Path, root: Path, raw_readme: Any) -> str:
    """Optionally write ``to_folder/README.md``. Returns a human-readable
    summary of what happened (``"skipped"``, ``"left existing ..."``, or
    ``"wrote ..."``)."""
    if not isinstance(raw_readme, str) or not raw_readme.strip():
        return "skipped"
    readme_path = to_folder / "README.md"
    if readme_path.exists():
        return f"left existing {readme_path.relative_to(root)} untouched"
    readme_path.write_text(raw_readme, encoding="utf-8")
    return f"wrote {readme_path.relative_to(root)} ({len(raw_readme)} chars)"


def _perform_moves(
    moves: list[tuple[Path, Path]],
    root: Path,
) -> tuple[list[str] | None, Dict[str, Any] | None]:
    """Execute the planned moves on disk. Returns ``(moved_paths, None)``
    on success or ``(None, err_response)`` on a partial-failure I/O error.
    Read-write-unlink (not shutil.move) so a partial failure is detectable
    and encoding stays under our control."""
    moved_paths: List[str] = []
    for src, dst in moves:
        if src == dst:
            moved_paths.append(f"{src.relative_to(root)} (already at destination)")
            continue
        try:
            content = src.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return None, _move_err(f"could not read {src.relative_to(root)}: {e}")
        dst.write_text(content, encoding="utf-8")
        src.unlink()
        moved_paths.append(f"{src.relative_to(root)} → {dst.relative_to(root)}")
    return moved_paths, None


def _rewrite_inbound_wikilinks(
    root: Path,
    link_mapping: Dict[str, str],
) -> tuple[int, list[str]]:
    """Rewrite every inbound wikilink in the library to its new target.
    Returns ``(total_rewrites, per_file_summary)``."""
    rewrite_summary: List[str] = []
    total_rewrites = 0
    for md in sorted(root.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_text, n = _rewrite_wikilinks_in_text(text, link_mapping)
        if n > 0:
            md.write_text(new_text, encoding="utf-8")
            rewrite_summary.append(f"{md.relative_to(root)}: {n} link(s) rewritten")
            total_rewrites += n
    return total_rewrites, rewrite_summary


def _build_move_summary(
    to_folder: Path,
    root: Path,
    readme_action: str,
    moved_paths: list[str],
    total_rewrites: int,
    rewrite_summary: list[str],
) -> Dict[str, Any]:
    lines = ["move_entries: ok"]
    lines.append(f"  destination: {to_folder.relative_to(root)}/")
    lines.append(f"  readme: {readme_action}")
    lines.append(f"  moved {len(moved_paths)} file(s):")
    for m in moved_paths:
        lines.append(f"    - {m}")
    lines.append(
        f"  rewrote {total_rewrites} inbound wikilink(s)"
        + (":" if rewrite_summary else " (no inbound links found)")
    )
    for r in rewrite_summary:
        lines.append(f"    - {r}")
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


def _move_entries_impl(root: Path, args: Dict[str, Any]) -> Dict[str, Any]:
    """Pure-Python implementation of move_entries — see tool docstring.

    Validates inputs, performs the moves, rewrites inbound wikilinks
    everywhere in the library, and returns a textual summary the LLM
    can read.
    """
    raw_to = str(args.get("to_folder") or "").strip()
    raw_files = args.get("files")
    raw_readme = args.get("readme")
    # The SDK passes typed-list args through as Python lists already.
    if not raw_to:
        return _move_err("missing required arg 'to_folder'")
    if not isinstance(raw_files, list) or not raw_files:
        return _move_err("'files' must be a non-empty list of .md paths")

    to_folder, err = _resolve_destination(root, raw_to)
    if err is not None:
        return err
    assert to_folder is not None  # narrowed

    moves, link_mapping, err = _plan_moves(root, to_folder, cast(List[object], raw_files))
    if err is not None:
        return err
    assert moves is not None  # narrowed

    to_folder.mkdir(parents=True, exist_ok=True)
    readme_action = _maybe_write_readme(to_folder, root, raw_readme)

    moved_paths, err = _perform_moves(moves, root)
    if err is not None:
        return err
    assert moved_paths is not None  # narrowed

    total_rewrites, rewrite_summary = _rewrite_inbound_wikilinks(root, link_mapping)

    return _build_move_summary(
        to_folder, root, readme_action, moved_paths, total_rewrites, rewrite_summary
    )
