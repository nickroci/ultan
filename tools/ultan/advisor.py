#!/usr/bin/env python3
"""Ultan advisor — ask the knowledge library for advice on a question.

Invoked by the `/ultan-advisor` Claude Code slash command. Two-step
pipeline:

  1. Librarian (Sonnet, read-only): explores the library — BM25 search,
     Read, Glob — and returns the most-relevant entries plus a short
     "what's here that matters" summary.
  2. Scholar (Opus): takes the question + the Librarian's findings and
     writes a referenced answer with wikilinks to the cited entries.

Output is markdown printed to stdout. Claude Code's slash command
expansion shows it inline in the user's transcript.

Stdlib + claude-agent-sdk only. Reuses the daemon's
``agent_mem_daemon.library_tools.make_library_mcp_server`` so BM25 is
the same in both places. Pure read — never writes to disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def home() -> Path:
    env = os.environ.get("AGENT_MEM_HOME")
    return Path(env).expanduser().resolve() if env else (Path.home() / ".agent-mem")


def knowledge_dir() -> Path:
    return home() / "knowledge"


# Keep model names in sync with daemon/agent_mem_daemon/config.py — importing
# the daemon's config eagerly would pull the heavy embeddings stack onto the
# --help path, so this carries its own copy instead.
LIBRARIAN_MODEL = "claude-sonnet-4-6"
SCHOLAR_MODEL = "claude-opus-4-8"

# Shorter than the daemon's 600s: the advisor is interactive (a user is
# waiting on /ultan-advisor), not a background batch.
LIBRARIAN_TIMEOUT_S = 300.0
SCHOLAR_TIMEOUT_S = 300.0


# ── Librarian prompt — find relevant entries ────────────────────────


_LIBRARIAN_PROMPT = """You are the Ultan Librarian in ADVISOR MODE.

The user is about to make a decision, write code, or ask a question. They want to know what the library already remembers that's relevant. Your job is to FIND THE RIGHT ENTRIES — you are not the one answering, the Scholar does that next.

USER QUESTION:
  {question}

LIBRARY LAYOUT:
  - Root: knowledge/ — top-level structure has global/ and projects/<slug>/.
  - Each entry is a markdown file with YAML frontmatter (id, type, scope, title, applies-when, ...).
  - Each folder has a README.md with a `<!-- ULTAN:children (auto) -->` block listing children.

TOOLS YOU HAVE:
  - mcp__agent_mem_library__bm25_search(query, k) — best tool for "is there anything about X?" Returns top-K relevant entries with snippets.
  - Glob(pattern) — find files by name pattern (e.g. `**/python/**/*.md`).
  - Grep(pattern, path) — literal regex over file contents.
  - Read(file_path) — fetch a specific entry's full body.

PROCEDURE:
  1. Start with ``bm25_search`` on key terms from the question. Try multiple phrasings — 2-3 searches is normal. Each search returns the top hits with snippets.
  2. Read the most promising hits in full so you can judge relevance and pull out the key claim.
  3. If you suspect a topic-folder might cover the question, Glob for it (e.g. `Glob("**/deployment/**/*.md")`).
  4. Stop when you've found at most 5 highly-relevant entries — more is noise.

OUTPUT (your FINAL message — single JSON object, no fences, no commentary):

{{
  "relevant_entries": [
    {{
      "path": "<path relative to knowledge/>",
      "title": "<frontmatter title>",
      "key_claim": "<one sentence: what this entry says about the question>",
      "directly_addresses": true|false
    }}
  ],
  "notes_for_scholar": "<one short paragraph: what's in the library that bears on the question, and what's NOT there. Be honest if the library has nothing relevant — empty is a valid finding.>"
}}

If the library has nothing relevant: ``{{"relevant_entries": [], "notes_for_scholar": "Library has no entries on this topic."}}``. The Scholar will say so honestly.
"""


# ── Scholar prompt — synthesise the answer ──────────────────────────


_SCHOLAR_PROMPT = """You are the Ultan Scholar in ADVISOR MODE.

You are a LIBRARIAN, not a consultant. The user asked a question. The Librarian searched the shelves and gathered the relevant entries. Your job: report what's in the library, accurately and concisely. That is the entire job.

USER QUESTION:
  {question}

LIBRARIAN'S FINDINGS:
  {librarian_findings}

RULES:
  - Be concise. Default to 3-6 sentences. Bullet-list 2-5 points if structure helps.
  - **Reference every claim** that comes from a stored entry with a wikilink: ``[[path]]`` (no .md). Inline, where the claim is made.
  - Distinguish what's stored from what isn't:
      • Stored library knowledge → cited with [[wikilink]]
      • If you must offer a baseline-knowledge fill-in (only when the library is silent and the user clearly needs *some* answer), mark it "(not from memory)". Keep it to a single sentence.
  - If the library has nothing relevant, say so plainly: "The library has no entries on this." Do not pad.
  - If a stored entry directly contradicts the user's premise, quote it. Otherwise stay neutral.

DO NOT:
  - Suggest next steps, follow-ups, or actions for the calling agent to take.
  - Suggest restructuring the library, promoting entries, adding hooks, blocking tools, or any other meta-advice about Ultan itself. The background curator handles that.
  - Editorialize ("you should…", "I'd recommend…", "consider…"). Report, don't prescribe.
  - Repeat the same point in different words to look thorough.

OUTPUT FORMAT:
  Plain markdown. No preamble like "Here's my advice:". Start with what the library says.
  Use the wikilink format ``[[global/python/use-uv-not-pip]]`` (no extension).
"""


# ── SDK driver ──────────────────────────────────────────────────────


async def _prompt_stream(text: str):
    yield {"type": "user", "message": {"role": "user", "content": text}}


async def _drain_query(prompt: str, options) -> str:
    """Run one streaming query and return concatenated text. Tool-use
    blocks are NOT inlined — we only care about the final text response."""
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        query,
    )

    out = ""
    async for msg in query(prompt=_prompt_stream(prompt), options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    out += block.text
        elif isinstance(msg, ResultMessage):
            pass  # cost tracked elsewhere
    return out


def _make_path_guard(boundary: Path):
    """Read-only path guard: only Read/Glob/Grep + the BM25 MCP tool."""
    root = boundary.expanduser().resolve()
    PATH_KEYS = {"Read": ["file_path"], "Glob": ["path"], "Grep": ["path"]}
    PATH_FREE = {"mcp__agent_mem_library__bm25_search"}

    async def _can_use_tool(name, input_, ctx):  # noqa: ARG001
        from claude_agent_sdk.types import (
            PermissionResultAllow,
            PermissionResultDeny,
        )

        if name in PATH_FREE:
            return PermissionResultAllow(updated_input=input_)
        if name not in PATH_KEYS:
            return PermissionResultDeny(message=f"tool {name!r} not allowed for the advisor")
        for key in PATH_KEYS[name]:
            raw = input_.get(key)
            if raw is None:
                continue
            try:
                Path(str(raw)).expanduser().resolve().relative_to(root)
            except (ValueError, OSError):
                return PermissionResultDeny(message=f"path {raw!r} is outside {root}")
        return PermissionResultAllow(updated_input=input_)

    return _can_use_tool


async def _run_librarian(question: str, kdir: Path) -> dict:
    """Invoke the Librarian to find relevant entries. Returns a parsed
    JSON dict, or a safe default if parsing fails."""
    # Lazy imports: missing SDK must not kill --help, and the daemon's
    # library_tools transitively pulls the heavy embeddings stack — keep it off
    # the import-time / --help path. library_tools reuses the daemon's BM25 MCP
    # server so search behaves identically in both places.
    from agent_mem_daemon import library_tools
    from claude_agent_sdk import ClaudeAgentOptions

    options = ClaudeAgentOptions(
        model=LIBRARIAN_MODEL,
        cwd=str(kdir),
        allowed_tools=[
            "Read",
            "Glob",
            "Grep",
            library_tools.fully_qualified_tool_name(),
        ],
        mcp_servers={
            library_tools.SERVER_NAME: library_tools.make_library_mcp_server(kdir),
        },
        max_turns=15,
        system_prompt={"type": "preset", "preset": "claude_code"},
        can_use_tool=_make_path_guard(kdir),
        env={**os.environ, "CLAUDE_INVOKED_BY": "ultan_advisor"},
    )
    text = await asyncio.wait_for(
        _drain_query(_LIBRARIAN_PROMPT.format(question=question), options),
        timeout=LIBRARIAN_TIMEOUT_S,
    )
    return _parse_librarian_json(text)


async def _run_scholar(question: str, findings: dict, kdir: Path) -> str:
    """Invoke the Scholar to synthesise the advice. Returns markdown."""
    from claude_agent_sdk import ClaudeAgentOptions

    findings_json = json.dumps(findings, indent=2, ensure_ascii=False)
    prompt = _SCHOLAR_PROMPT.format(
        question=question,
        librarian_findings=findings_json,
    )
    options = ClaudeAgentOptions(
        model=SCHOLAR_MODEL,
        cwd=str(kdir),
        # Scholar is also read-only in advisor mode — no Write/Edit.
        # It just composes the answer from the Librarian's findings.
        allowed_tools=["Read"],
        max_turns=5,
        system_prompt={"type": "preset", "preset": "claude_code"},
        can_use_tool=_make_path_guard(kdir),
        env={**os.environ, "CLAUDE_INVOKED_BY": "ultan_advisor"},
    )
    return (
        await asyncio.wait_for(
            _drain_query(prompt, options),
            timeout=SCHOLAR_TIMEOUT_S,
        )
    ).strip()


def _parse_librarian_json(text: str) -> dict:
    """Best-effort JSON parse — falls back to a safe-empty default.

    The advisor doesn't need perfect parsing; the Scholar can handle a
    Librarian who just dumped a paragraph instead of JSON.
    """
    text = text.strip()
    # Strip a possible fenced block.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].lstrip().startswith("```") else lines[1:])
    # Find the last top-level {...} (same trick as the daemon's parser).
    start = -1
    depth = 0
    blocks: list[str] = []
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if esc:
            esc = False
            continue
        if in_str:
            if c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start : i + 1])
                start = -1
    if not blocks:
        return {
            "relevant_entries": [],
            "notes_for_scholar": "(Librarian returned no parseable JSON; raw text: "
            + text[:500]
            + ")",
        }
    try:
        parsed = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return {
            "relevant_entries": [],
            "notes_for_scholar": "(Librarian JSON malformed; raw text: " + blocks[-1][:500] + ")",
        }
    if not isinstance(parsed, dict):
        return {"relevant_entries": [], "notes_for_scholar": str(parsed)[:500]}
    parsed.setdefault("relevant_entries", [])
    parsed.setdefault("notes_for_scholar", "")
    return parsed


# ── Entry point ─────────────────────────────────────────────────────


def run(question: str) -> int:
    """Ask Ultan for advice on ``question``. Shared entry point for both the
    ``advisor.py`` CLI and the ``ultan advisor`` subcommand.

    Runs the Librarian (find) → Scholar (synthesise) pipeline and prints the
    referenced answer to stdout. Read-only — never writes to the library.
    """
    question = question.strip()
    if not question:
        print("ultan-advisor: empty question", file=sys.stderr)
        return 2

    kdir = knowledge_dir()
    if not kdir.exists():
        print(f"(library not initialised yet at {kdir} — nothing to advise from)")
        return 0

    try:
        print("_Ultan: searching library..._", flush=True)
        findings = asyncio.run(_run_librarian(question, kdir))
        n_hits = len(findings.get("relevant_entries") or [])
        print(
            f"_Ultan: found {n_hits} relevant entr"
            f"{'y' if n_hits == 1 else 'ies'}, synthesising..._",
            flush=True,
        )
        answer = asyncio.run(_run_scholar(question, findings, kdir))
    except asyncio.TimeoutError:
        print("ultan-advisor: timed out", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ultan-advisor: failed: {e}", file=sys.stderr)
        return 1

    print()
    print(answer)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ask Ultan for advice. Searches the knowledge library and "
        "synthesises a referenced answer.",
    )
    ap.add_argument(
        "question", nargs="*", help="The question or decision. Use '-' to read from stdin."
    )
    args = ap.parse_args()

    if args.question == ["-"] or not args.question:
        if sys.stdin.isatty() and not args.question:
            ap.error("no question supplied (pass it as args, or pipe via '-')")
        question = sys.stdin.read()
    else:
        question = " ".join(args.question)

    return run(question)


if __name__ == "__main__":
    raise SystemExit(main())
