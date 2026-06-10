---
name: ultan-search
description: Search and traverse the user's Ultan memory library (~/.agent-mem/knowledge/). Use whenever you see an Ultan wikilink in priming context (e.g. [[projects/foo/bar]]) and want to read it, OR when you want to find entries by topic. Accepts a wikilink/path (returns the entry plus directory context — siblings, subfolders, parent README) or a free-text query (returns top-K matches with snippets). Cheaper than multiple raw Read calls because it bundles content with the surrounding hierarchy in one shot. Use this in preference to Read for any path under ~/.agent-mem/knowledge/.
---

# Ultan Search

Talks to the running `agent-mem-daemon` over its Unix socket and returns either a single entry (with browseable neighbours) or a ranked list of matches.

## Usage

Invoke with **one argument** — the script auto-detects mode:

- **Wikilink or path** — `[[projects/agent-mem/concepts/README]]`, `projects/agent-mem/concepts/README`, `global/conventions/use-uv.md`, etc. → returns the entry's full content **plus** siblings, subfolders, and parent README excerpt so you can navigate to neighbours without a second call.
- **Free-text query** — `optimizer pruning`, `tier 3 interrupts`, `prefer python over javascript`, etc. → returns top-8 BM25-ranked entries with score + snippet.

Explicit prefixes win if auto-detection is wrong: `path:<x>` or `query:<x>`.

## Running it

Run the `search.py` that sits in this skill's own directory (the directory
containing this SKILL.md — shown in the skill listing):

```bash
python "<this-skill-dir>/search.py" "<wikilink-or-query>"
```

Use `Bash` to run this — the script writes the result as markdown to stdout.
It is stdlib-only; any `python3` works.

## When NOT to use this skill

- You want a **synthesised** answer fusing multiple entries — use `/ultan-advisor <question>` instead. The advisor runs Sonnet (deep search) + Opus (synthesis) to produce a single referenced answer; this skill is the raw-fetch primitive.
- The priming snippet you already received in `additionalContext` covers the question — don't drill down redundantly.
- You're trying to *write* to the library — use `/ultan <text>` or let the curator capture it from conversation; this skill is read-only.

## Daemon dependency

This skill requires the agent-mem daemon to be running. If the socket isn't
present at `${AGENT_MEM_HOME:-~/.agent-mem}/priming.sock`, the script prints a
clear error. With the Ultan plugin installed the daemon lazy-starts on its own
— retry after a turn or two; from source, start it with `uv run agent-mem-daemon -v`.
