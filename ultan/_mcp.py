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

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _kick_provisioner() -> None:
    """Fire the plugin's background installer if the runtime isn't provisioned.

    The plugin spec has no install-time script, but Claude Code starts this
    MCP server right after `/plugin install` + `/reload-plugins` — the
    earliest code the plugin gets to run — so kicking here makes provisioning
    start at install time in practice. Idempotent and cheap: ensure-ultan.sh
    returns immediately and holds an atomic install lock, so overlapping
    kicks from the MCP server, SessionStart, and the first-prompt fallback
    all collapse into one install. plugin.json passes CLAUDE_PLUGIN_ROOT /
    CLAUDE_PLUGIN_DATA into our env; missing vars degrade to a no-op (the
    other triggers still cover provisioning)."""
    data = os.environ.get("CLAUDE_PLUGIN_DATA") or str(
        Path.home() / ".claude" / "plugins" / "data" / "ultan-ultan"
    )
    if (Path(data) / "bin" / "ultan").exists():
        return  # already provisioned
    root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if not root:
        return
    script = Path(root) / "scripts" / "ensure-ultan.sh"
    if not script.exists():
        return
    try:
        subprocess.Popen(
            ["bash", str(script)],
            env={**os.environ, "CLAUDE_PLUGIN_DATA": data},
            stdout=subprocess.DEVNULL,  # the installer logs to install.log itself
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        pass


def build_server() -> "FastMCP":
    """Construct the FastMCP server and register tools. Pure construction — no
    daemon spawn, no I/O — so it's unit-testable."""
    from mcp.server.fastmcp import FastMCP

    from . import _daemon, _priming

    server = FastMCP("ultan")

    @server.tool()
    def ultan_recall(query: str) -> str:  # pyright: ignore[reportUnusedFunction]  # registered via decorator
        """Recall relevant lessons, preferences, and conventions from the
        user's Ultan memory library for the given query. Returns markdown
        wikilink bullets (open them with the Read tool) or a no-match note."""
        result = _priming.get_priming(query, k=5) or "(no relevant Ultan memory for this query)"
        # Fallback honesty: during warmup results come from the lexical scan,
        # and a "no match" may just mean the ranked index isn't serving yet.
        if _daemon.status() == "warming":
            return (
                "*(Ultan daemon is warming up — lexical-fallback results; "
                "retry in a minute or two for full ranked recall.)*\n\n" + result
            )
        return result

    return server


def serve() -> int:
    """Run the MCP server over stdio (what Claude Code launches). Kicks the
    plugin provisioner if the runtime is missing (closest thing to an
    install-time hook — see _kick_provisioner), then async-spawns the daemon;
    neither blocks the MCP handshake or Claude Code's server-startup timeout."""
    from . import _daemon

    _kick_provisioner()
    _daemon.ensure_running()
    build_server().run()
    return 0
