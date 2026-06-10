"""Ultan CLI — the `ultan` entry point.

Installed via `uv tool install git+https://github.com/nickroci/ultan` (the Claude
Code plugin provisions it on first use via scripts/ensure-ultan.sh). Subcommands:
`hook` (the per-turn hot path — stdlib-light and torch-free, guarded by
tests/test_hook_import.py), `daemon` (runs the agent-mem-daemon installed in this
same venv), `mcp` (the light MCP server Claude Code launches), and `advisor` /
`remember` (the /ultan-advisor and /ultan slash commands). The heavy ML deps are
lazy-imported so they never touch the hook path. `ultan doctor` reports whether
the retrieval stack (agent-mem-search) resolved.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main() -> int:
    ap = argparse.ArgumentParser(prog="ultan", description="Ultan — local memory for Claude Code.")
    ap.add_argument("--version", action="store_true", help="Print version and exit.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="Report install / dependency-resolution status.")
    d = sub.add_parser("daemon", help="Run the memory daemon installed in this venv.")
    d.add_argument(
        "daemon_args", nargs=argparse.REMAINDER, help="Args forwarded to the daemon (e.g. -v)."
    )
    h = sub.add_parser("hook", help="Run a Claude Code hook handler (talks to the daemon).")
    h.add_argument("event", help="Hook event, e.g. user-prompt-submit, session-start.")
    sub.add_parser("mcp", help="Run the light MCP server (Claude Code launches this).")
    adv = sub.add_parser(
        "advisor", help="Ask the library for advice (the /ultan-advisor slash command)."
    )
    adv.add_argument("query", nargs="*", help="The question or decision to advise on.")
    rem = sub.add_parser(
        "remember", help="Queue a memory for the Librarian (the /ultan slash command)."
    )
    rem.add_argument("text", nargs="*", help="The memory text to remember.")
    rem.add_argument(
        "--global",
        dest="globally",
        action="store_true",
        help="Mark this memory as global (cross-project) scope.",
    )
    rem.add_argument("--scope", help="Explicit project slug override.")
    args = ap.parse_args()

    if args.version:
        print(f"ultan {__version__}")
        return 0
    if args.cmd == "doctor":
        from . import _doctor

        return _doctor.run()
    if args.cmd == "daemon":
        from . import _daemon

        return _daemon.run_foreground(args.daemon_args)
    if args.cmd == "hook":
        from . import _hooks

        return _hooks.dispatch(args.event)
    if args.cmd == "mcp":
        from . import _mcp

        return _mcp.serve()
    # advisor / remember ship via the [retrieval] extra (agent-mem-tools), so
    # a thin install — including the root CI job's env — deliberately lacks
    # them. stubs/*.pyi give pyright their signatures either way, so these
    # call sites type identically in thin and full envs.
    if args.cmd == "advisor":
        # Lazy: advisor pulls claude-agent-sdk + the daemon's library_tools
        # (heavy embeddings stack). Must stay off the hook hot path.
        import advisor

        return advisor.run(" ".join(args.query))
    if args.cmd == "remember":
        # Lazy: keeps the stdlib-only remember module off the hook import path
        # too, for symmetry and a clean hot path.
        import remember

        return remember.run(" ".join(args.text), globally=args.globally, scope=args.scope)

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
