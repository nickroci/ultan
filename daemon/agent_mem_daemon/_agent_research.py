"""Read-only research tools shared by the two curator roles, exposed to the
model through the Claude Agent SDK.

In the typed-output architecture the Librarian and the Scholar both run via
``typed_agent.run_typed`` on ``claude_agent_sdk`` (the subscription backend —
never the metered API). Each role returns a *typed* result and WRITES NOTHING
to disk; a deterministic executor applies the Scholar's actions. So the only
tools the model needs are READ-ONLY research tools:

  - ``read_entry``      — read one knowledge file by knowledge-relative path.
  - ``grep_library``    — case-insensitive literal substring search.
  - ``bm25_search``     — lexical (BM25) relevance search.
  - ``embedding_search``— semantic (embedding) similarity search.

These mirror exactly the four ``@agent.tool`` read-only tools the reverted
Pydantic-AI migration registered via ``_agent_common.register_research_tools``;
here they are re-expressed as SDK ``@tool``s in an in-process
``create_sdk_mcp_server`` so no model-write tool is ever wired. The
knowledge-store root is captured at server-construction time, so the model
cannot inject a path outside the store.

``research_server_and_tools(knowledge_dir)`` returns the ``(server, dict_key,
allowed_tool_names)`` triple a role hands straight to ``run_typed`` as
``mcp_servers`` / ``allowed_tools``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from . import library_tools

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpServerConfig

log = logging.getLogger("agent_mem_daemon._agent_research")

# Server + tool naming. Keep stable; the ``allowed_tools`` names the roles
# pass to ``run_typed`` are derived from these.
SERVER_NAME = "agent_mem_research"
_READ_ENTRY_TOOL = "read_entry"
_GREP_TOOL = "grep_library"
_BM25_TOOL = "bm25_search"
_EMBEDDING_TOOL = "embedding_search"


def _tool_ref(name: str) -> str:
    """Fully-qualified ``allowed_tools`` name for a research tool."""
    return f"mcp__{SERVER_NAME}__{name}"


def research_tool_refs() -> List[str]:
    """The fully-qualified names of every read-only research tool, for a
    role's ``allowed_tools`` list."""
    return [
        _tool_ref(_READ_ENTRY_TOOL),
        _tool_ref(_GREP_TOOL),
        _tool_ref(_BM25_TOOL),
        _tool_ref(_EMBEDDING_TOOL),
    ]


# ── Pure tool helpers (module-level, easy to unit-test without a model) ──────


def inside(root: Path, candidate: Path) -> bool:
    """True iff ``candidate`` resolves inside ``root`` (no path escape)."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def read_entry(knowledge_dir: Path, path: str) -> str:
    """Read a knowledge file by its knowledge-relative path. Returns the
    contents, or a ``(not found ...)`` / refusal sentinel. Read-only."""
    root = knowledge_dir.resolve()
    target = (root / path).resolve()
    if not inside(root, target):
        return f"(path {path!r} resolves outside the knowledge store — refused)"
    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"(not found: {path})"
    except OSError as e:
        return f"(could not read {path}: {e})"


def grep_library(knowledge_dir: Path, pattern: str, path: str) -> str:
    """Case-insensitive literal-substring search over the library, optionally
    scoped to a subdirectory. Returns up to 40 ``<rel>:<line-no>: <line>``
    matches, a truncation note past that, or a sentinel. Read-only,
    ``_archive`` excluded."""
    root = knowledge_dir.resolve()
    if not pattern.strip():
        return "(grep_library: empty pattern)"
    scope = (root / path).resolve() if path else root
    if not inside(root, scope) or not scope.exists():
        return f"(grep_library: {path!r} not found under the knowledge store)"
    needle = pattern.lower()
    search_root = scope if scope.is_dir() else scope.parent
    out: List[str] = []
    for md in sorted(search_root.rglob("*.md")):
        if "_archive" in md.parts:
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                rel = md.relative_to(root).as_posix()
                out.append(f"{rel}:{i}: {line.strip()}")
                if len(out) >= 40:
                    return "\n".join(out) + "\n(truncated at 40 matches)"
    return "\n".join(out) if out else f"(no matches for {pattern!r})"


def search_text(
    runner: Callable[[Dict[str, Any], Path], Dict[str, Any]],
    knowledge_dir: Path,
    query: str,
    k: int,
) -> str:
    """Call a library search runner and unwrap its MCP-shaped response into
    the plain text the model reads."""
    response = runner({"query": query, "k": k}, knowledge_dir.resolve())
    text = library_tools.unwrap_text_response(response)
    return text or "(search returned no content)"


# ── SDK research-tool server ─────────────────────────────────────────────────


def make_research_mcp_server(knowledge_dir: Path) -> "McpServerConfig":
    """Build an in-process SDK MCP server exposing only the four read-only
    research tools, scoped to ``knowledge_dir``.

    There is NO file-writing tool here — the model returns a typed result and
    a deterministic executor is the only writer. The root is captured at
    construction time so the model can never inject a path outside the store.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: PLC0415

    root = knowledge_dir.expanduser().resolve()

    @tool(
        _READ_ENTRY_TOOL,
        (
            "Read a knowledge file by its path relative to the knowledge root "
            "(e.g. `index.md` or `global/python/use-uv.md`). Returns the file "
            "contents, or a `(not found ...)` sentinel. Read-only."
        ),
        {"path": str},
    )
    async def read_entry_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        return library_tools.text_response(read_entry(root, str(args.get("path") or "")))

    @tool(
        _GREP_TOOL,
        (
            "Search the knowledge library for a literal substring "
            "(case-insensitive), optionally scoped to a subdirectory `path`. "
            "Returns up to 40 `<rel-path>:<line-no>: <line>` matches. "
            "Read-only."
        ),
        {"pattern": str, "path": str},
    )
    async def grep_library_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        return library_tools.text_response(
            grep_library(root, str(args.get("pattern") or ""), str(args.get("path") or ""))
        )

    @tool(
        _BM25_TOOL,
        (
            "Lexical (BM25) search over the library. Returns the top-K entries "
            "as `<path>  score=<float>  <snippet>` lines. Complements "
            "`embedding_search` — run both for any concept query. Read-only. "
            "Typical k 3-8."
        ),
        {"query": str, "k": int},
    )
    async def bm25_search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        k = _coerce_k(args.get("k"))
        return library_tools.text_response(
            search_text(library_tools.run_bm25_search, root, str(args.get("query") or ""), k)
        )

    @tool(
        _EMBEDDING_TOOL,
        (
            "Semantic (embedding) search over the library. Returns the top-K "
            "entries as `<path>  score=<float>  <snippet>` lines. Complements "
            "`bm25_search` — run both for any concept query. Read-only. "
            "Typical k 3-8."
        ),
        {"query": str, "k": int},
    )
    async def embedding_search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        k = _coerce_k(args.get("k"))
        return library_tools.text_response(
            search_text(library_tools.run_embedding_search, root, str(args.get("query") or ""), k)
        )

    # Reference the closures so linters don't flag them as unused — the
    # decorator already registered each on the server.
    _ = (read_entry_tool, grep_library_tool, bm25_search_tool, embedding_search_tool)

    return create_sdk_mcp_server(
        name=SERVER_NAME,
        version="1.0.0",
        tools=[read_entry_tool, grep_library_tool, bm25_search_tool, embedding_search_tool],
    )


def _coerce_k(raw: object) -> int:
    """Lenient ``k`` coercion: default 6, clamped to [1, 20]."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return 6
    try:
        k = int(raw)
    except (TypeError, ValueError):
        k = 6
    return max(1, min(20, k))


def research_server_and_tools(
    knowledge_dir: Path,
) -> Tuple[Dict[str, "McpServerConfig"], List[str]]:
    """Return ``(mcp_servers, allowed_tools)`` for a role, ready to splat into
    :func:`typed_agent.run_typed`. ``mcp_servers`` carries the read-only
    research server under :data:`SERVER_NAME`; ``allowed_tools`` lists the four
    research tools (``run_typed`` adds the ``submit_result`` tool itself)."""
    server = make_research_mcp_server(knowledge_dir)
    return {SERVER_NAME: server}, research_tool_refs()
