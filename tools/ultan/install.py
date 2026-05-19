#!/usr/bin/env python3
"""Install agent-mem hooks into a project's `.claude/settings.json`.

Invoked by the `/ultan-install` slash command (or directly from a shell).
Stdlib-only — no install/sync step.

What it does:
  1. Resolves the template at <repo>/src/_dot_claude_disabled/settings.json.
  2. Substitutes AGENT_MEM_SRC -> the absolute src path.
  3. Reads the project's existing .claude/settings.json (or {}).
  4. Merges the `hooks` block in (overwriting agent-mem hooks if present,
     preserving any other settings).
  5. Writes back atomically.
  6. Prints what changed and a note about starting the daemon.

Default target: <cwd>/.claude/settings.json. Override with --target <path>
or --global to write into ~/.claude/settings.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
TEMPLATE = SRC / "_dot_claude_disabled" / "settings.json"

AGENT_MEM_HOOK_EVENTS = {
    "SessionStart", "UserPromptSubmit", "PostToolUse",
    "Stop", "PreCompact", "SessionEnd",
}


def load_template() -> dict:
    raw = TEMPLATE.read_text(encoding="utf-8")
    raw = raw.replace("AGENT_MEM_SRC", str(SRC))
    data = json.loads(raw)
    # Drop the metadata keys.
    return {"hooks": data["hooks"]}


def merge_hooks(existing: dict, template_hooks: dict) -> tuple[dict, list[str]]:
    """Return (merged_settings, list_of_event_names_changed).

    For each event in the template:
      - If existing has no hooks for that event, add ours.
      - If existing has hooks, we replace any entry that looks like an
        agent-mem hook (its command contains the agent-mem src path) and
        append ours if none matched. Non-agent-mem hooks are preserved.
    """
    merged = dict(existing)
    hooks = dict(merged.get("hooks") or {})
    changed: list[str] = []

    for event, ours_list in template_hooks.items():
        ours_entry = ours_list[0]  # template has exactly one entry per event
        existing_event = hooks.get(event)
        if not existing_event:
            hooks[event] = [ours_entry]
            changed.append(event)
            continue

        # Walk existing entries; replace any agent-mem-shaped command.
        new_event = []
        for entry in existing_event:
            inner = entry.get("hooks") or []
            kept = [h for h in inner if str(SRC) not in str(h.get("command", ""))]
            if not kept:
                # Whole entry was just agent-mem hooks; drop it and let
                # ours replace it below.
                continue
            if len(kept) == len(inner):
                new_event.append(entry)
            else:
                new_event.append({**entry, "hooks": kept})

        new_event.append(ours_entry)
        # Only mark changed if the effective list actually differs.
        if new_event != existing_event:
            changed.append(event)
        hooks[event] = new_event

    merged["hooks"] = hooks
    return merged, changed


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Install agent-mem hooks into a project's .claude/settings.json."
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--target", type=Path,
                   help="Path to settings.json (defaults to <cwd>/.claude/settings.json).")
    g.add_argument("--global", dest="globally", action="store_true",
                   help="Install at ~/.claude/settings.json (every project).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change; don't write.")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        print(f"ultan-install: template not found at {TEMPLATE}", file=sys.stderr)
        return 2

    if args.globally:
        target = Path.home() / ".claude" / "settings.json"
    elif args.target:
        target = args.target.expanduser().resolve()
    else:
        target = (Path.cwd() / ".claude" / "settings.json").resolve()

    template = load_template()
    existing: dict = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"ultan-install: existing {target} is not valid JSON: {e}",
                  file=sys.stderr)
            return 2
        if not isinstance(existing, dict):
            print(f"ultan-install: existing {target} is not a JSON object",
                  file=sys.stderr)
            return 2

    merged, changed = merge_hooks(existing, template["hooks"])

    if args.dry_run:
        print(f"target: {target}")
        print(f"events that would change: {changed or '(none)'}")
        print("--- merged settings would be ---")
        print(json.dumps(merged, indent=2))
        return 0

    write_atomic(target, merged)
    print(f"installed hooks at {target}")
    print(f"events written: {sorted(AGENT_MEM_HOOK_EVENTS)}")
    if changed:
        print(f"events changed by this install: {changed}")
    else:
        print("(no effective change — hooks were already installed)")
    print()
    print("To run the daemon (foreground, tail-friendly):")
    print(f"  cd {REPO / 'daemon'} && uv run agent-mem-daemon -v")
    print()
    print("Or in the background:")
    print(f"  cd {REPO / 'daemon'} && \\")
    print(f"    nohup uv run agent-mem-daemon -v > ~/.agent-mem/daemon.stdout 2>&1 &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
