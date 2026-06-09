"""Ultan CLI (thin wrapper).

Experiment scaffold for branch experiment/uv-tool-install: proves the thin
`ultan` entry point installs from git via `uv tool install`, and that
`ultan doctor` can detect whether the heavy `retrieval` extra (the workspace
sibling agent-mem-search) was resolved and installed alongside it.

Real subcommands (install / uninstall / daemon / hook) come once the
packaging path is validated.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

from . import __version__


def _retrieval_available() -> bool:
    """True if the heavy retrieval stack (agent-mem-search) resolved and
    installed. `bm25` is a top-level module in the agent-mem-search wheel."""
    return importlib.util.find_spec("bm25") is not None


def main() -> int:
    ap = argparse.ArgumentParser(prog="ultan", description="Ultan — local memory for Claude Code.")
    ap.add_argument("--version", action="store_true", help="Print version and exit.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="Report install / dependency-resolution status.")
    d = sub.add_parser("daemon", help="Run the memory daemon (provisioned via uvx).")
    d.add_argument(
        "daemon_args", nargs=argparse.REMAINDER, help="Args forwarded to the daemon (e.g. -v)."
    )
    args = ap.parse_args()

    if args.version:
        print(f"ultan {__version__}")
        return 0
    if args.cmd == "doctor":
        print(f"ultan {__version__}")
        print(f"python: {sys.version.split()[0]}")
        print(f"retrieval extra (agent-mem-search) available: {_retrieval_available()}")
        return 0
    if args.cmd == "daemon":
        from . import _daemon

        return _daemon.run_foreground(args.daemon_args)

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
