#!/usr/bin/env bash
# CI guard (run by ci.yml on every PR): the plugin version, every workspace
# package version, and the install-spec refs must agree — so a release can never
# ship a plugin.json that points at the wrong code, or a half-bumped workspace.
#
# Coherent == ALL of:
#   - every workspace pyproject [project] version equals plugin.json "version"
#   - the `nickroci/ultan@<ref>` install spec in BOTH plugin.json and
#     scripts/ensure-ultan.sh is either "main" (unreleased / dev between
#     releases) or exactly "v<that version>".
#
# scripts/set-version.sh keeps these in lockstep; this is the backstop against a
# manual edit drifting one of them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

plugin_ver="$(perl -ne 'if (/"version":\s*"([^"]+)"/) { print $1; exit }' "$ROOT/.claude-plugin/plugin.json")"
[ -n "$plugin_ver" ] || { echo "FAIL: could not read plugin.json version"; exit 1; }
echo "plugin.json version: $plugin_ver"

for rel in pyproject.toml daemon/pyproject.toml tools/search/pyproject.toml tools/ultan/pyproject.toml src/pyproject.toml; do
  v="$(perl -ne 'if (/^version = "([^"]+)"/) { print $1; exit }' "$ROOT/$rel")"
  if [ "$v" != "$plugin_ver" ]; then
    echo "FAIL: $rel version '$v' != plugin.json '$plugin_ver'"; fail=1
  fi
done

for rel in .claude-plugin/plugin.json scripts/ensure-ultan.sh; do
  ref="$(perl -ne 'if (m{nickroci/ultan\@([A-Za-z0-9._/-]+)}) { print $1; exit }' "$ROOT/$rel")"
  if [ -z "$ref" ]; then
    echo "FAIL: no nickroci/ultan@<ref> install spec found in $rel"; fail=1
  elif [ "$ref" != "main" ] && [ "$ref" != "v$plugin_ver" ]; then
    echo "FAIL: $rel install spec '@$ref' is neither '@main' nor '@v$plugin_ver'"; fail=1
  fi
done

if [ "$fail" -eq 0 ]; then echo "version coherence OK"; fi
exit "$fail"
