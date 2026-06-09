#!/usr/bin/env bash
# SessionStart provisioner for the Ultan plugin (claude-mem-style hybrid).
#
# Ensures the THIN `ultan` CLI is installed into the plugin's PERSISTENT data
# dir (${CLAUDE_PLUGIN_DATA}), so the per-turn hooks can call the installed
# binary directly (no per-prompt uvx overhead). Then warms the daemon in the
# background. Fast on every run after the first; never blocks the session.
#
# The MCP server is provisioned separately via `uvx` (see plugin.json), so it
# does not depend on this script having run first.
set -euo pipefail

DATA="${CLAUDE_PLUGIN_DATA:?CLAUDE_PLUGIN_DATA is not set}"
BIN_DIR="$DATA/bin"
BIN="$BIN_DIR/ultan"
# TODO(release): pin to a published version / tag instead of a branch.
SPEC="git+https://github.com/nickroci/ultan@experiment/uv-tool-install"

if ! command -v uv >/dev/null 2>&1; then
  echo "ultan: 'uv' not found on PATH — install uv (https://docs.astral.sh/uv) to enable Ultan." >&2
  exit 0 # never block the session
fi

if [ ! -x "$BIN" ]; then
  # First run: install the thin CLI (light — pyyaml + mcp, no torch) into the
  # plugin data dir. The heavy retrieval stack is provisioned later, on demand,
  # when the daemon first starts.
  if ! UV_TOOL_BIN_DIR="$BIN_DIR" UV_TOOL_DIR="$DATA/uv-tools" \
    uv tool install --force "$SPEC" >/dev/null 2>&1; then
    echo "ultan: failed to install the CLI via 'uv tool install'." >&2
    exit 0
  fi
fi

# Warm the daemon in the background via the lazy-spawn path. Reuses
# `ultan hook session-start` -> ensure_running(); detached, never blocks.
"$BIN" hook session-start </dev/null >/dev/null 2>&1 &
exit 0
