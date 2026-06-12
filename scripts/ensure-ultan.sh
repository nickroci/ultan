#!/usr/bin/env bash
# SessionStart provisioner for the Ultan plugin (option B: background install).
#
# Returns IMMEDIATELY so it never hits the SessionStart hook timeout. The full
# `ultan[retrieval]` install pulls the heavy retrieval stack (torch etc.) and
# can take minutes on a cold cache, so it runs DETACHED in the background.
# Until it finishes, the per-turn hooks no-op gracefully (they guard on the
# binary existing — see hooks/hooks.json). Once installed, the daemon is warmed.
#
# The MCP server is provisioned separately via `uvx` (see plugin.json) and
# stays on the thin base install (no extra) — it does not depend on this script.
#
# Never block the session: every failure path exits 0 with at most a one-line
# stderr note (hence no `set -e` — each fallible step is guarded explicitly).
set -uo pipefail

DATA="${CLAUDE_PLUGIN_DATA:-}"
if [ -z "$DATA" ]; then
  echo "ultan: CLAUDE_PLUGIN_DATA is not set — cannot provision." >&2
  exit 0
fi
BIN_DIR="$DATA/bin"
BIN="$BIN_DIR/ultan"
LOCK="$DATA/.install.lock"
# The install spec. scripts/set-version.sh rewrites the `@<ref>` here to the
# release tag (`@vX.Y.Z`) at release time; between releases it tracks `@main`.
# ULTAN_SPEC override lets scripts/validate-plugin.sh install the local tree.
SPEC="${ULTAN_SPEC:-ultan[retrieval] @ git+https://github.com/nickroci/ultan@v0.3.2}"

# The git ref the SPEC pins (the @<ref> at the very end) and the uv receipt that
# records what is actually installed — together they drive the up-to-date check
# below. TARGET_REF is empty when SPEC has no @ref (e.g. a local-path dev
# override), in which case we never auto-update.
TARGET_REF="$(printf '%s' "$SPEC" | sed -n 's/.*@\([A-Za-z0-9._/-]*\)$/\1/p')"
RECEIPT="$DATA/uv-tools/ultan/uv-receipt.toml"

if ! command -v uv >/dev/null 2>&1; then
  echo "ultan: 'uv' not found on PATH — install uv (https://docs.astral.sh/uv) to enable Ultan." >&2
  exit 0
fi

mkdir -p "$BIN_DIR" 2>/dev/null || {
  echo "ultan: cannot create $BIN_DIR — skipping install." >&2
  exit 0
}

# Health probe: an executable shim whose venv interpreter is gone (interrupted
# reinstall, moved/upgraded python) would otherwise silently no-op forever —
# `[ -x "$BIN" ]` alone can never notice. Cheap: read the shebang, stat it.
if [ -x "$BIN" ]; then
  interp="$(sed -n '1s/^#!//p' "$BIN" 2>/dev/null)"
  if [ -n "$interp" ] && [ ! -x "$interp" ]; then
    rm -f "$BIN" 2>/dev/null || true # broken shim — fall through to reinstall
  fi
fi

# Already installed: decide warm-and-return vs update. If the installed git ref
# matches the SPEC's target ref (or we can't tell), just warm the daemon and
# return. If it's stale — a newer release tag is pinned — fall through to an
# INCREMENTAL update (no --force) so uv swaps only the changed packages and
# reuses the heavy venv + warm cache (seconds, not a full reinstall).
UPDATE=""
if [ -x "$BIN" ] && [ -n "$TARGET_REF" ] && [ -f "$RECEIPT" ]; then
  CURRENT_REF="$(sed -n 's/.*[?&]rev=\([^"&]*\).*/\1/p' "$RECEIPT" | head -1)"
  if [ -n "$CURRENT_REF" ] && [ "$CURRENT_REF" != "$TARGET_REF" ]; then
    UPDATE=1
  fi
fi
if [ -x "$BIN" ] && [ -z "$UPDATE" ]; then
  ( nohup "$BIN" hook session-start </dev/null >/dev/null 2>&1 & )
  exit 0
fi

# Missing, broken, or stale. Take the install lock ATOMICALLY (noclobber `set -C` create
# fails iff the file exists — no check-then-create race). If the create fails,
# an install is already running — unless the lock is stale (>30 min: a
# previous install died), in which case clear it and retry the create once.
if ! ( set -C; : >"$LOCK" ) 2>/dev/null; then
  if find "$LOCK" -mmin -30 2>/dev/null | grep -q .; then
    exit 0
  fi
  rm -f "$LOCK" 2>/dev/null || exit 0
  ( set -C; : >"$LOCK" ) 2>/dev/null || exit 0
fi
# Stamp the lock with an ownership token: only the install that wrote it may
# clear it, so a finishing install can't delete a peer's fresh lock and let a
# third session pile on a concurrent `uv tool install`.
TOKEN="$$-$(date +%s)"
printf '%s' "$TOKEN" >"$LOCK" 2>/dev/null || { rm -f "$LOCK" 2>/dev/null; exit 0; }

# Detached background install — orphaned to init via `( nohup … & )` so it
# survives this hook returning. Env vars are exported so the inner shell
# inherits them (avoids fragile quoting). On success, warm the daemon; clear
# the lock (if still ours) so a failed attempt retries next session. The
# installer records its PID in .install.pid so validate-plugin.sh can wait
# for / kill it.
# A fresh or broken install gets a clean --force; a stale-version update does
# NOT, so uv reuses the existing venv and only swaps the changed packages
# (verified: a changed spec without --force updates in place from cache).
if [ -n "$UPDATE" ]; then UV_FORCE=""; else UV_FORCE="--force"; fi
export _ULTAN_BIN="$BIN" _ULTAN_BIN_DIR="$BIN_DIR" _ULTAN_DATA="$DATA" \
  _ULTAN_LOCK="$LOCK" _ULTAN_SPEC="$SPEC" _ULTAN_TOKEN="$TOKEN" _ULTAN_FORCE="$UV_FORCE"
( nohup bash -c '
    echo "$$" >"$_ULTAN_DATA/.install.pid" 2>/dev/null || true
    if UV_TOOL_BIN_DIR="$_ULTAN_BIN_DIR" UV_TOOL_DIR="$_ULTAN_DATA/uv-tools" \
         uv tool install $_ULTAN_FORCE "$_ULTAN_SPEC" >"$_ULTAN_DATA/install.log" 2>&1; then
      "$_ULTAN_BIN" hook session-start </dev/null >/dev/null 2>&1 || true
    fi
    rm -f "$_ULTAN_DATA/.install.pid" 2>/dev/null || true
    if [ "$(cat "$_ULTAN_LOCK" 2>/dev/null)" = "$_ULTAN_TOKEN" ]; then
      rm -f "$_ULTAN_LOCK"
    fi
  ' >/dev/null 2>&1 & )

exit 0
