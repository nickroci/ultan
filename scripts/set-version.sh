#!/usr/bin/env bash
# Set the release version across the whole workspace in one shot — the single
# source of truth for "what version is everything". Run by the release workflow
# (.github/workflows/release.yml) and usable locally:
#
#   bash scripts/set-version.sh 0.3.0
#
# It rewrites, in place:
#   - [project] version in every workspace pyproject.toml (root + members)
#   - "version" in .claude-plugin/plugin.json
#   - the git ref of the `git+https://github.com/nickroci/ultan@<ref>` install
#     spec in plugin.json AND scripts/ensure-ultan.sh  ->  @v<version>
#   - uv.lock (re-locked so `uv sync --locked` stays valid — the member version
#     bump otherwise leaves the lock stale and reddens CI on main)
#
# It does NOT commit, tag, or push. The caller does that, so the version edits
# always land in the commit the tag will point at (the plugin.json install spec
# is a static literal Claude Code reads as-is, so it must self-reference the very
# tag being cut — edit, THEN commit, THEN tag).
set -euo pipefail

VERSION="${1:?usage: set-version.sh X.Y.Z}"
printf '%s' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
  || { echo "error: version must be X.Y.Z (got '$VERSION')" >&2; exit 1; }
TAG="v$VERSION"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Workspace members (root first). Keep in sync with [tool.uv.workspace] members
# in the root pyproject.toml + scripts/check-version-coherence.sh.
PYPROJECTS=(
  pyproject.toml
  daemon/pyproject.toml
  tools/search/pyproject.toml
  tools/ultan/pyproject.toml
)

for rel in "${PYPROJECTS[@]}"; do
  V="$VERSION" perl -0pi -e 's/^version = "[^"]*"/version = "$ENV{V}"/m' "$ROOT/$rel"
done

# plugin.json "version"
V="$VERSION" perl -0pi -e 's/("version":\s*")[^"]*(")/$1$ENV{V}$2/' "$ROOT/.claude-plugin/plugin.json"

# install-spec git ref -> @vX.Y.Z, in both the MCP manifest and the provisioner.
# (Matches a literal git ref after `nickroci/ultan@`; valid-ref chars only, so
# the trailing `}"` / `"` around the spec is left intact.)
TAG="$TAG" perl -0pi -e 's{(nickroci/ultan)\@[A-Za-z0-9._/-]+}{$1\@$ENV{TAG}}g' \
  "$ROOT/.claude-plugin/plugin.json" "$ROOT/scripts/ensure-ultan.sh"

# Re-lock: the workspace-member version bumps make uv.lock stale, so
# `uv sync --locked` (CI) would fail on main after the release merges. uv lock
# only rewrites the version strings here (no dependency change), and resolves
# from cache in milliseconds. Guarded so a uv-less local run still does the edits.
if command -v uv >/dev/null 2>&1; then
  ( cd "$ROOT" && uv lock )
else
  echo "warning: uv not found — skipped 'uv lock'; run it before committing or CI will fail" >&2
fi

echo "set version -> $VERSION  (install spec ref -> $TAG)"
