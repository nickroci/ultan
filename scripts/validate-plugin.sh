#!/usr/bin/env bash
# Programmatic validation of the Ultan Claude Code plugin install path.
#
# Validates everything that does NOT require a live, interactive Claude Code
# session. The final `/plugin install` + in-session hook/MCP wiring still needs
# a human CC session — see docs / the validation report for the manual guide.
#
# What this checks (all in throwaway TEMP dirs — never touches your live
# ~/.claude or ~/.agent-mem):
#   1. Manifests are valid JSON (plugin.json, marketplace.json, hooks.json).
#   2. The SessionStart provisioner (scripts/ensure-ultan.sh) installs the thin
#      `ultan` CLI into ${CLAUDE_PLUGIN_DATA}/bin and the binary runs.
#   3. The thin install is genuinely thin (mcp + pyyaml, NO torch).
#   4. The MCP server completes a JSON-RPC initialize + tools/list handshake
#      over stdio (proving Claude Code can launch and talk to it).
#   5. The UserPromptSubmit hook exits 0 and emits additionalContext JSON
#      (or nothing, gracefully, when no library exists on disk).
#
# Usage:  bash scripts/validate-plugin.sh
# Exit:   0 if all checks pass, 1 otherwise.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL + 1)); }
info() { printf '  ....  %s\n' "$1"; }
section() { printf '\n== %s ==\n' "$1"; }

# --- Isolated temp dirs (mandatory: never the live install) -----------------
export AGENT_MEM_HOME="$(mktemp -d "${TMPDIR:-/tmp}/ultan-val-home.XXXXXX")"
export CLAUDE_PLUGIN_DATA="$(mktemp -d "${TMPDIR:-/tmp}/ultan-val-data.XXXXXX")"
export CLAUDE_PLUGIN_ROOT="$REPO_ROOT"
BIN="$CLAUDE_PLUGIN_DATA/bin/ultan"

cleanup() {
  # Kill only daemons tied to OUR temp home; leave any other daemon alone.
  pkill -f "AGENT_MEM_HOME=$AGENT_MEM_HOME" 2>/dev/null || true
  rm -rf "$AGENT_MEM_HOME" "$CLAUDE_PLUGIN_DATA" 2>/dev/null || true
}
trap cleanup EXIT

printf 'Ultan plugin validation\n'
printf 'CLAUDE_PLUGIN_ROOT=%s\n' "$CLAUDE_PLUGIN_ROOT"
printf 'CLAUDE_PLUGIN_DATA=%s (temp)\n' "$CLAUDE_PLUGIN_DATA"
printf 'AGENT_MEM_HOME=%s (temp)\n' "$AGENT_MEM_HOME"

# --- Check 1: manifests are valid JSON --------------------------------------
section "1. Manifest JSON validity"
for f in .claude-plugin/plugin.json .claude-plugin/marketplace.json hooks/hooks.json; do
  if python3 -m json.tool "$REPO_ROOT/$f" >/dev/null 2>&1; then
    ok "$f is valid JSON"
  else
    bad "$f is NOT valid JSON"
  fi
done

# --- Preconditions ----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  bad "uv not found on PATH — cannot run provisioner / MCP / hook checks"
  printf '\nResult: %d passed, %d failed\n' "$PASS" "$FAIL"
  exit 1
fi

# --- Check 2: provisioner installs the thin CLI -----------------------------
section "2. Provisioner (scripts/ensure-ultan.sh)"
if bash "$REPO_ROOT/scripts/ensure-ultan.sh"; then
  if [ -x "$BIN" ]; then
    ver="$("$BIN" --version 2>/dev/null || true)"
    if [ -n "$ver" ]; then
      ok "installed CLI runs: $ver"
    else
      bad "binary exists but --version produced no output"
    fi
  else
    bad "expected binary not found at \$CLAUDE_PLUGIN_DATA/bin/ultan"
  fi
else
  bad "ensure-ultan.sh exited non-zero"
fi

# --- Check 3: thin install has no torch -------------------------------------
section "3. Thin install (no torch)"
SP="$(find "$CLAUDE_PLUGIN_DATA/uv-tools" -maxdepth 4 -name site-packages -type d 2>/dev/null | head -1)"
if [ -n "$SP" ]; then
  if ls "$SP" | grep -qiE '^(torch|sentence_transformers)'; then
    bad "heavy deps (torch/sentence_transformers) leaked into the thin install"
  else
    ok "no torch/sentence_transformers in thin env"
  fi
  if ls "$SP" | grep -qi '^mcp' && ls "$SP" | grep -qi '^yaml'; then
    ok "thin env has mcp + pyyaml (expected runtime deps)"
  else
    bad "thin env missing expected mcp/pyyaml"
  fi
else
  bad "could not locate the installed env's site-packages"
fi

# --- Check 4: MCP stdio handshake -------------------------------------------
section "4. MCP server stdio handshake"
MCP_OUT="$(AGENT_MEM_HOME="$AGENT_MEM_HOME" python3 - <<'PY' 2>/dev/null
import json, os, subprocess, sys, threading, time
proc = subprocess.Popen(
    ["uvx", "--from", os.environ["CLAUDE_PLUGIN_ROOT"], "ultan", "mcp"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1, env=dict(os.environ))
def send(o): proc.stdin.write(json.dumps(o) + "\n"); proc.stdin.flush()
out = []
threading.Thread(target=lambda: [out.append(l.strip()) for l in proc.stdout if l.strip()], daemon=True).start()
time.sleep(3)
send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
    "protocolVersion":"2025-06-18","capabilities":{},
    "clientInfo":{"name":"validate","version":"0"}}})
time.sleep(1.5)
send({"jsonrpc":"2.0","method":"notifications/initialized"})
time.sleep(0.5)
send({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
time.sleep(2)
proc.terminate()
try: proc.wait(timeout=5)
except subprocess.TimeoutExpired: proc.kill()
print("\n".join(out))
PY
)"
if printf '%s' "$MCP_OUT" | grep -q '"serverInfo"' && printf '%s' "$MCP_OUT" | grep -q '"protocolVersion"'; then
  ok "initialize handshake returned a valid result"
else
  bad "initialize handshake did not return a valid result"
fi
if printf '%s' "$MCP_OUT" | grep -q 'ultan_recall'; then
  ok "tools/list advertises the ultan_recall tool"
else
  bad "tools/list did not advertise ultan_recall"
fi

# --- Check 5: UserPromptSubmit hook -----------------------------------------
section "5. UserPromptSubmit hook"
# 5a: empty library -> exit 0, no output
OUT_EMPTY="$(echo '{"prompt":"how do I use uv","session_id":"t"}' | "$BIN" hook user-prompt-submit 2>/dev/null)"
RC=$?
if [ "$RC" -eq 0 ] && [ -z "$OUT_EMPTY" ]; then
  ok "no-library case: exit 0 and emits nothing (graceful)"
else
  bad "no-library case: rc=$RC output='${OUT_EMPTY:0:60}'"
fi
# 5b: seed a tiny library -> expect additionalContext JSON
KDIR="$AGENT_MEM_HOME/knowledge"; mkdir -p "$KDIR"
cat > "$KDIR/uv-tips.md" <<'MD'
---
title: Using uv for Python
applies-when: |
  Installing or running Python tools with uv / uvx
keywords: [uv, uvx, python, install, tool]
reinforced: 2
---
# Using uv
Use `uv tool install` for CLIs and `uvx` for one-shot runs.
MD
OUT_SEED="$(echo '{"prompt":"how do I use uv to install a python tool","session_id":"t"}' | "$BIN" hook user-prompt-submit 2>/dev/null)"
RC=$?
if [ "$RC" -eq 0 ] && printf '%s' "$OUT_SEED" | grep -q '"additionalContext"' && printf '%s' "$OUT_SEED" | grep -q 'UserPromptSubmit'; then
  ok "seeded-library case: exit 0 and emits additionalContext JSON"
else
  bad "seeded-library case: rc=$RC output='${OUT_SEED:0:80}'"
fi

# --- Summary ----------------------------------------------------------------
printf '\n== Summary ==\n'
printf 'Passed: %d   Failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
