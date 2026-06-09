#!/usr/bin/env bash
# Programmatic validation of the Ultan Claude Code plugin install path.
#
# Validates everything that does NOT require a live, interactive Claude Code
# session. The final `/plugin install` + in-session hook/MCP wiring still needs
# a human CC session — see docs / the validation report for the manual guide.
#
# What this checks (all in throwaway TEMP dirs — never touches your live
# ~/.claude or ~/.agent-mem):
#   1. Manifests are valid JSON *and structurally sound* (top-level `hooks`
#      key in hooks.json; plugin.json must NOT reference the auto-loaded
#      hooks/commands dirs — that caused the 'Duplicate hooks file' regression).
#   2. The SessionStart provisioner (scripts/ensure-ultan.sh) installs the
#      LOCAL TREE (via the ULTAN_SPEC override) into ${CLAUDE_PLUGIN_DATA}/bin
#      and the binary runs. The install is detached, so this WAITS for it
#      (cold cache pulls torch — set ULTAN_VALIDATE_TIMEOUT, default 900s).
#   3. The provisioned env is the FULL runtime: `ultan[retrieval]` puts the
#      agent-mem-daemon console script + retrieval stack in the same venv
#      (the thin no-extra install is the MCP/uvx path, exercised by check 4).
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
WAIT_S="${ULTAN_VALIDATE_TIMEOUT:-900}"

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL + 1)); }
info() { printf '  ....  %s\n' "$1"; }
section() { printf '\n== %s ==\n' "$1"; }

# --- Isolated temp dirs (mandatory: never the live install) -----------------
export AGENT_MEM_HOME="$(mktemp -d "${TMPDIR:-/tmp}/ultan-val-home.XXXXXX")"
export CLAUDE_PLUGIN_DATA="$(mktemp -d "${TMPDIR:-/tmp}/ultan-val-data.XXXXXX")"
export CLAUDE_PLUGIN_ROOT="$REPO_ROOT"
# Validate the working tree, not whatever is on remote main.
export ULTAN_SPEC="ultan[retrieval] @ file://$REPO_ROOT"
BIN="$CLAUDE_PLUGIN_DATA/bin/ultan"

cleanup() {
  # Kill the detached installer if it is still running (it records its PID).
  if [ -f "$CLAUDE_PLUGIN_DATA/.install.pid" ]; then
    kill "$(cat "$CLAUDE_PLUGIN_DATA/.install.pid" 2>/dev/null)" 2>/dev/null || true
  fi
  # Kill only processes tied to OUR temp dirs. pkill -f matches argv: the
  # daemon's argv is the console-script path under $CLAUDE_PLUGIN_DATA, so
  # match on that (env vars like AGENT_MEM_HOME never appear in argv).
  pkill -f "$CLAUDE_PLUGIN_DATA" 2>/dev/null || true
  sleep 0.5
  rm -rf "$AGENT_MEM_HOME" "$CLAUDE_PLUGIN_DATA" 2>/dev/null || true
}
trap cleanup EXIT

printf 'Ultan plugin validation\n'
printf 'CLAUDE_PLUGIN_ROOT=%s\n' "$CLAUDE_PLUGIN_ROOT"
printf 'CLAUDE_PLUGIN_DATA=%s (temp)\n' "$CLAUDE_PLUGIN_DATA"
printf 'AGENT_MEM_HOME=%s (temp)\n' "$AGENT_MEM_HOME"
printf 'ULTAN_SPEC=%s\n' "$ULTAN_SPEC"

# --- Check 1: manifests are valid JSON and structurally sound ----------------
section "1. Manifests (syntax + structure)"
for f in .claude-plugin/plugin.json .claude-plugin/marketplace.json hooks/hooks.json; do
  if python3 -m json.tool "$REPO_ROOT/$f" >/dev/null 2>&1; then
    ok "$f is valid JSON"
  else
    bad "$f is NOT valid JSON"
  fi
done
# hooks.json: events must live under a single top-level "hooks" key (the
# missing-wrapper bug fixed in ff25022 was valid JSON and slipped through).
if python3 - "$REPO_ROOT/hooks/hooks.json" <<'PY' >/dev/null 2>&1
import json, sys
h = json.load(open(sys.argv[1]))
assert set(h) == {"hooks"}, f"top-level keys: {set(h)}"
events = h["hooks"]
assert isinstance(events, dict) and events, "hooks must be a non-empty object"
assert all(isinstance(v, list) for v in events.values()), "each event maps to a list"
assert {"SessionStart", "UserPromptSubmit"} <= set(events), "core events missing"
PY
then
  ok "hooks.json structure: top-level 'hooks' wrapper + core events"
else
  bad "hooks.json structure invalid (missing 'hooks' wrapper or core events)"
fi
# plugin.json: must NOT reference auto-loaded dirs ('Duplicate hooks file',
# fixed in 33bfe64 — also valid JSON, also slipped through).
if python3 - "$REPO_ROOT/.claude-plugin/plugin.json" <<'PY' >/dev/null 2>&1
import json, sys
p = json.load(open(sys.argv[1]))
assert "hooks" not in p and "commands" not in p, "auto-loaded dirs must not be referenced"
PY
then
  ok "plugin.json does not re-reference auto-loaded hooks/commands"
else
  bad "plugin.json references auto-loaded hooks/commands (Duplicate hooks file)"
fi

# --- Preconditions ----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  bad "uv not found on PATH — cannot run provisioner / MCP / hook checks"
  printf '\nResult: %d passed, %d failed\n' "$PASS" "$FAIL"
  exit 1
fi

# --- Check 2: provisioner installs the CLI (wait for the detached install) ---
section "2. Provisioner (scripts/ensure-ultan.sh — installs the local tree)"
if bash "$REPO_ROOT/scripts/ensure-ultan.sh"; then
  ok "ensure-ultan.sh returned immediately (exit 0)"
else
  bad "ensure-ultan.sh exited non-zero"
fi
info "waiting for the detached install (timeout ${WAIT_S}s; cold cache pulls torch)…"
t=0
while [ -f "$CLAUDE_PLUGIN_DATA/.install.lock" ] && [ "$t" -lt "$WAIT_S" ]; do
  sleep 5; t=$((t + 5))
  if [ $((t % 60)) -eq 0 ]; then
    info "still installing (${t}s): $(tail -n 1 "$CLAUDE_PLUGIN_DATA/install.log" 2>/dev/null | cut -c1-80)"
  fi
done
if [ -x "$BIN" ]; then
  ver="$("$BIN" --version 2>/dev/null || true)"
  if [ -n "$ver" ]; then
    ok "installed CLI runs: $ver"
  else
    bad "binary exists but --version produced no output"
  fi
else
  bad "binary not at \$CLAUDE_PLUGIN_DATA/bin/ultan after ${t}s (see install.log)"
  info "install.log tail: $(tail -n 3 "$CLAUDE_PLUGIN_DATA/install.log" 2>/dev/null | tr '\n' ' | ')"
fi

# --- Check 3: provisioned env is the FULL runtime ----------------------------
section "3. Full runtime env (ultan[retrieval]: daemon + retrieval stack)"
SP="$(find "$CLAUDE_PLUGIN_DATA/uv-tools" -maxdepth 4 -name site-packages -type d 2>/dev/null | head -1)"
ENV_BIN="$(find "$CLAUDE_PLUGIN_DATA/uv-tools" -maxdepth 3 -name agent-mem-daemon -type f 2>/dev/null | head -1)"
if [ -n "$SP" ]; then
  # Direct -d tests, NOT `ls | grep -q`: grep -q's early exit can SIGPIPE ls
  # under pipefail and silently invert the verdict.
  if [ -d "$SP/torch" ] && [ -d "$SP/sentence_transformers" ]; then
    ok "retrieval stack (torch + sentence_transformers) present"
  else
    bad "retrieval stack missing — [retrieval] extra did not install"
  fi
  if [ -d "$SP/mcp" ]; then
    ok "mcp (base dep, for 'ultan mcp') present"
  else
    bad "mcp missing from the provisioned env"
  fi
  if [ -n "$ENV_BIN" ] && [ -x "$ENV_BIN" ]; then
    ok "agent-mem-daemon console script present (same-venv daemon spawn works)"
  else
    bad "agent-mem-daemon console script missing — ultan/_daemon.py cannot spawn"
  fi
else
  bad "could not locate the installed env's site-packages"
fi

# --- Check 4: MCP stdio handshake (thin uvx path) ----------------------------
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
if [ -x "$BIN" ]; then
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
else
  bad "skipping hook checks — binary was never installed (see check 2)"
fi

# --- Summary ----------------------------------------------------------------
printf '\n== Summary ==\n'
printf 'Passed: %d   Failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
