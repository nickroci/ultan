#!/usr/bin/env bash
# SessionStart provisioner for the Ultan plugin (option B: background install).
#
# Returns IMMEDIATELY so it never hits the SessionStart hook timeout. The full
# `ultan` install pulls the heavy retrieval stack (torch etc.) and can take
# minutes on a cold cache, so it runs DETACHED in the background. Until it
# finishes, the per-turn hooks no-op gracefully (they guard on the binary
# existing — see hooks/hooks.json). Once installed, the daemon is warmed.
#
# The MCP server is provisioned separately via `uvx` (see plugin.json), so it
# does not depend on this script.
set -euo pipefail

DATA="${CLAUDE_PLUGIN_DATA:?CLAUDE_PLUGIN_DATA is not set}"
BIN_DIR="$DATA/bin"
BIN="$BIN_DIR/ultan"
LOCK="$DATA/.install.lock"
# TODO(release): pin to a published version / tag instead of a branch.
SPEC="git+https://github.com/nickroci/ultan@main"

if ! command -v uv >/dev/null 2>&1; then
  echo "ultan: 'uv' not found on PATH — install uv (https://docs.astral.sh/uv) to enable Ultan." >&2
  exit 0 # never block the session
fi

mkdir -p "$BIN_DIR"

# Already installed: warm the daemon (detached) and return immediately.
if [ -x "$BIN" ]; then
  ( nohup "$BIN" hook session-start </dev/null >/dev/null 2>&1 & )
  exit 0
fi

# Not installed. Skip if a background install is already running (lock < 30 min
# old); a staler lock means a previous install died — clear it and retry.
if [ -f "$LOCK" ]; then
  if find "$LOCK" -mmin -30 2>/dev/null | grep -q .; then
    exit 0
  fi
  rm -f "$LOCK"
fi
: >"$LOCK"

# Detached background install — orphaned to init via `( nohup … & )` so it
# survives this hook returning. Env vars are exported so the inner shell
# inherits them (avoids fragile quoting). On success, warm the daemon; always
# clear the lock so a failed attempt retries next session.
export _ULTAN_BIN="$BIN" _ULTAN_BIN_DIR="$BIN_DIR" _ULTAN_DATA="$DATA" \
  _ULTAN_LOCK="$LOCK" _ULTAN_SPEC="$SPEC"
( nohup bash -c '
    if UV_TOOL_BIN_DIR="$_ULTAN_BIN_DIR" UV_TOOL_DIR="$_ULTAN_DATA/uv-tools" \
         uv tool install --force "$_ULTAN_SPEC" >"$_ULTAN_DATA/install.log" 2>&1; then
      "$_ULTAN_BIN" hook session-start </dev/null >/dev/null 2>&1 || true
    fi
    rm -f "$_ULTAN_LOCK"
  ' >/dev/null 2>&1 & )

exit 0
