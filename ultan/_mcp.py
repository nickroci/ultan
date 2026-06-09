"""Light, fast-starting MCP server for the Ultan Claude Code plugin.

Claude Code launches this when the plugin is enabled and manages its lifecycle.
It MUST start quickly, so it does not load the heavy retrieval stack (no torch).
On startup it async-spawns the memory daemon — non-blocking, so the MCP
handshake never waits on the ~25s model boot — then exposes recall tools that
proxy to the daemon over the Unix socket, falling back to a crude stdlib
lexical scan while the daemon is still warming.

Keep this module's import light: the heavy work is the daemon's, behind the
socket. `mcp` is imported lazily inside the functions so it never lands on the
hook hot path (see tests/test_hook_import.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def build_server() -> "FastMCP":
    """Construct the FastMCP server and register tools. Pure construction — no
    daemon spawn, no I/O — so it's unit-testable."""
    from mcp.server.fastmcp import FastMCP

    from . import _priming

    server = FastMCP("ultan")

    @server.tool()
    def ultan_recall(query: str) -> str:
        """Recall relevant lessons, preferences, and conventions from the
        user's Ultan memory library for the given query. Returns markdown
        wikilink bullets (open them with the Read tool) or a no-match note."""
        return _priming.get_priming(query, k=5) or "(no relevant Ultan memory for this query)"

    return server


def serve() -> int:
    """Run the MCP server over stdio (what Claude Code launches). Async-spawns
    the daemon first so memory comes up in the background without blocking the
    MCP handshake or hitting Claude Code's server-startup timeout."""
    from . import _daemon

    _daemon.ensure_running()
    build_server().run()
    return 0
