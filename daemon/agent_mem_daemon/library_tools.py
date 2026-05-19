"""In-process MCP tools the Librarian (and optionally Scholar) can call.

Right now this exposes a single ``bm25_search`` tool that wraps the
existing ``agent-mem-search`` BM25 indexer. The Librarian uses it to
find entries semantically related to a phrase or concept — complements
the SDK's built-in ``Glob`` (filename pattern) and ``Grep`` (literal
regex) tools.

We register the tools as an SDK MCP server (``McpSdkServerConfig``) so
they run in the same Python process as the daemon — no subprocess, no
HTTP, no extra deps. The Librarian's prompt is told they exist; the
SDK's ``allowed_tools`` list whitelists the canonical
``mcp__<server>__<tool>`` name.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict


log = logging.getLogger("agent_mem_daemon.library_tools")


# Server + tool naming. The SDK exposes MCP tools to the model as
# ``mcp__<server_name>__<tool_name>``. Keep these stable; allowed_tools
# entries in llm.py refer to them.
_SERVER_NAME = "agent_mem_library"
_TOOL_NAME = "bm25_search"


def fully_qualified_tool_name() -> str:
    """The string to put in ``allowed_tools`` for the model to see the tool."""
    return f"mcp__{_SERVER_NAME}__{_TOOL_NAME}"


def make_library_mcp_server(knowledge_dir: Path):
    """Build and return an SDK MCP server config exposing BM25 search.

    Returns the value to pass into ``ClaudeAgentOptions.mcp_servers``
    under any dict key (the daemon uses ``_SERVER_NAME``). The
    knowledge_dir is captured at construction time so the tool always
    searches the same store — no path injection possible from the model.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    root = knowledge_dir.expanduser().resolve()

    @tool(
        _TOOL_NAME,
        (
            "Search the agent-mem knowledge library by content relevance "
            "(BM25 ranking). Returns the top-K most relevant entries as "
            "lines of `<path>  score=<float>  <one-line snippet>`. Use "
            "this when you suspect an entry already covers a topic but "
            "don't see it in the library snapshot — it complements Glob "
            "(filename pattern) and Grep (literal regex). Typical k is "
            "3-8; larger values just add noise on a small corpus."
        ),
        {"query": str, "k": int},
    )
    async def bm25_search(args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "").strip()
        try:
            k = int(args.get("k") or 5)
        except (TypeError, ValueError):
            k = 5
        if not query:
            return {
                "content": [
                    {"type": "text", "text": "(bm25_search: empty query)"}
                ]
            }

        try:
            from bm25 import load_or_build  # provided by agent-mem-search
        except ImportError as e:
            log.warning("bm25 package not importable: %s", e)
            return {
                "content": [
                    {"type": "text", "text": "(bm25 backend unavailable)"}
                ]
            }

        if not root.exists():
            return {
                "content": [
                    {"type": "text", "text": "(library is empty — no entries to search)"}
                ]
            }

        try:
            index = load_or_build(root)
        except FileNotFoundError:
            return {
                "content": [
                    {"type": "text", "text": "(library has no entries yet)"}
                ]
            }
        except Exception as e:
            log.exception("bm25 index load/build failed")
            return {
                "content": [
                    {"type": "text", "text": f"(bm25 backend error: {e})"}
                ]
            }

        hits = index.search(query, k=max(1, min(20, k)))
        if not hits:
            return {
                "content": [
                    {"type": "text", "text": f"(no results for {query!r})"}
                ]
            }

        lines = [
            f"{Path(p).relative_to(root) if Path(p).is_absolute() else p}  "
            f"score={score:.2f}  {snippet}"
            for p, score, snippet in hits
        ]
        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(lines),
                }
            ]
        }

    return create_sdk_mcp_server(
        name=_SERVER_NAME,
        version="1.0.0",
        tools=[bm25_search],
    )
