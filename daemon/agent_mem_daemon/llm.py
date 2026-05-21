"""Claude Agent SDK invocation wrappers shared by the Librarian and Scholar.

Both roles call ``claude_agent_sdk.query(...)`` but with different option
shapes. This module owns the SDK-touching code so the role modules stay
focused on prompt assembly and decision logic.

Two entry points:

- ``run_librarian_call(prompt, *, timeout_s)`` — text-only call, no tools,
  Haiku tier. Returns the concatenated text response and cost.

- ``run_scholar_call(prompt, *, cwd, timeout_s)`` — file-tool-enabled call,
  Opus tier, ``permission_mode="acceptEdits"``. Returns the concatenated
  text response and cost.

Both wrappers also set ``CLAUDE_INVOKED_BY`` in the spawned SDK process's
environment (via the ``env=`` option) so any user-installed hook can detect
the recursion-blocked context and bail. Same pattern as `flush.py`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Tuple

log = logging.getLogger("agent_mem_daemon.llm")


# Models. Originally Haiku for the Librarian, but Haiku-tier was
# under-extracting even textbook user preferences in live testing —
# the recall tier needs a stronger model. Sonnet is the right balance:
# cheaper than Opus, much more reliable than Haiku at "is this a
# preference worth capturing" judgement.
LIBRARIAN_MODEL = "claude-sonnet-4-6"
SCHOLAR_MODEL = "claude-opus-4-7"

# Both calls run on the daemon's background loop — never on the hot path
# of a user-facing agent turn. Generous timeouts are fine; we'd rather
# wait than drop a packet to a transient Haiku slow-call.
LIBRARIAN_TIMEOUT_S = 600.0
SCHOLAR_TIMEOUT_S = 600.0


class LLMTimeout(RuntimeError):
    pass


def _resolve_wikilink(link: str, knowledge_root: Path, file_being_written: Path) -> bool:
    """Same resolution rules as scholar_prompt.check_invariants. Returns
    True if the link resolves to an existing file under knowledge_root.
    """
    if link.startswith("_archive/") or "/_archive/" in link:
        return True
    if link.startswith("daily/"):
        return True
    if link.endswith("/"):
        target = knowledge_root / link / "README.md"
    else:
        target = knowledge_root / (link if link.endswith(".md") else f"{link}.md")
    if target.exists():
        return True
    # Sibling fallback — link is relative to the file's own dir.
    parent = file_being_written.parent if file_being_written.parent.exists() else knowledge_root
    if link.endswith("/"):
        sibling = parent / link / "README.md"
    else:
        sibling = parent / (link if link.endswith(".md") else f"{link}.md")
    return sibling.exists()


def _find_unique_leaf(leaf: str, knowledge_root: Path) -> str | None:
    """Look up a wikilink target by its leaf (final path segment).

    Returns the canonical wikilink target (knowledge-root-relative, no
    .md suffix) if exactly one .md file in the tree has that leaf;
    None otherwise (zero or multiple matches → ambiguous, don't repair).
    """
    if "/" in leaf:
        # Already a multi-segment path — caller wanted full-path lookup,
        # which we've already tried. Skip leaf-mode.
        return None
    if leaf.endswith(".md"):
        leaf = leaf[:-3]
    matches = list(knowledge_root.rglob(f"{leaf}.md"))
    matches = [m for m in matches if "_archive" not in m.parts]
    if len(matches) != 1:
        return None
    rel = matches[0].relative_to(knowledge_root)
    return rel.with_suffix("").as_posix()


def _check_and_repair_writes(
    tool_name: str,
    tool_input: dict,
    knowledge_root: Path,
) -> tuple[str, dict | None, str]:
    """Validate any wikilinks in a Write/Edit payload before it lands on
    disk. Returns a (status, payload, info) triple:

      - ("allow",             tool_input,  "")
            no changes needed.
      - ("allow_with_repair", new_input,   summary)
            broken links found, all auto-resolvable by unique-leaf
            lookup; ``new_input`` is the rewritten tool input.
      - ("deny",              None,        message)
            broken links found that we can't auto-fix.

    Skips log.md (audit-trail; quoted paths are not navigation).
    """
    from . import markdown_utils

    file_path = tool_input.get("file_path")
    if not file_path:
        return "allow", tool_input, ""
    target_file = Path(str(file_path)).expanduser().resolve()
    if target_file.name == "log.md":
        return "allow", tool_input, ""

    # What content is being placed on disk?
    if tool_name == "Write":
        content = tool_input.get("content")
        content_key = "content"
    elif tool_name == "Edit":
        content = tool_input.get("new_string")
        content_key = "new_string"
    else:
        return "allow", tool_input, ""
    if not isinstance(content, str) or not content.strip():
        return "allow", tool_input, ""

    hits = markdown_utils.extract_wikilinks(content)
    if not hits:
        return "allow", tool_input, ""

    repairs: list[tuple[str, str]] = []  # (broken, fixed)
    unresolvable: list[str] = []  # broken with no auto-fix

    for hit in hits:
        link = hit.target
        if not link:
            continue
        if _resolve_wikilink(link, knowledge_root, target_file):
            continue
        # Broken — try leaf-name lookup for auto-repair.
        leaf = link.rsplit("/", 1)[-1]
        canonical = _find_unique_leaf(leaf, knowledge_root)
        if canonical is None or canonical == link:
            unresolvable.append(link)
        else:
            repairs.append((link, canonical))

    if unresolvable:
        repair_note = ""
        if repairs:
            repair_note = (
                " (note: "
                + ", ".join(f"[[{b}]] → [[{f}]]" for b, f in repairs)
                + " would auto-resolve, but the unresolvable links above also "
                "need to be corrected before the write can proceed)"
            )
        msg = (
            f"Write to {target_file.name} rejected — these wikilinks do not "
            f"resolve and no unique leaf-name match was found in the library: "
            + ", ".join(f"[[{u}]]" for u in unresolvable)
            + repair_note
            + ". Use Glob/Grep/bm25_search to locate the intended target, "
            "or remove the link if the entry doesn't exist yet."
        )
        return "deny", None, msg

    if repairs:
        new_content = content
        for broken, fixed in repairs:
            new_content = _rewrite_link_in_text(new_content, broken, fixed)
        new_input = dict(tool_input)
        new_input[content_key] = new_content
        summary = ", ".join(f"[[{b}]] → [[{f}]]" for b, f in repairs)
        return "allow_with_repair", new_input, summary

    return "allow", tool_input, ""


def _rewrite_link_in_text(text: str, broken: str, fixed: str) -> str:
    """Rewrite `[[broken]]` and `[[broken|alias]]` to use ``fixed`` while
    preserving any alias. Mirrors library_tools._rewrite_wikilinks_in_text
    but for a single (broken, fixed) pair."""
    import re as _re

    # Match `[[broken]]` or `[[broken|alias]]` or `[[broken.md]]` (with
    # alias). Escape the broken target for regex use.
    escaped = _re.escape(broken)
    pattern = _re.compile(r"\[\[" + escaped + r"(?:\.md)?(\|[^\[\]]*)?\]\]")
    return pattern.sub(lambda m: f"[[{fixed}{m.group(1) or ''}]]", text)


def _make_path_guard(boundary: Path, *, allow_writes: bool, check_wikilinks: bool = False):
    """Return a ``can_use_tool`` callback that rejects any tool call
    referencing a path outside ``boundary``.

    This is INFRA-level enforcement — we do NOT trust the prompt to
    constrain paths. The Scholar previously read the user's real
    ``~/.agent-mem/`` even when given ``cwd=/tmp/ulttest`` because the
    LLM produced absolute paths. This callback prevents that class of
    bug entirely: any tool input whose path resolves outside the
    knowledge root is denied with a clear error.

    Args:
        boundary: the directory the role is allowed to touch. Any path
            that does not resolve inside this directory is denied.
        allow_writes: if False, Write/Edit are denied outright (read-only
            role like the Librarian). Read/Glob/Grep are always
            path-constrained.
        check_wikilinks: if True (Scholar), also parse the proposed
            content of every Write/Edit for wikilinks. Auto-repair
            broken-but-unique-leaf-resolvable links by mutating the tool
            input; deny writes that introduce unresolvable links.
            Catches the "Scholar references an old path after a move"
            class of bug before it lands on disk.
    """
    root = boundary.expanduser().resolve()

    # tool name → list of input keys that contain a filesystem path.
    PATH_KEYS = {
        "Read": ["file_path"],
        "Write": ["file_path"],
        "Edit": ["file_path"],
        "Glob": ["path"],
        "Grep": ["path"],
        "NotebookEdit": ["notebook_path"],
    }

    WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}

    # In-process MCP tools that don't take filesystem paths from the
    # model. They're safe by construction because the daemon pinned
    # their scope when it registered the server. Allow without path
    # validation.
    PATH_FREE_TOOLS = {
        "mcp__agent_mem_library__bm25_search",
        # move_entries validates every path against the knowledge root
        # internally (``_safe_inside``), so it's safe to allow without
        # the guard's per-key path validation.
        "mcp__agent_mem_library__move_entries",
    }

    async def _can_use_tool(tool_name, tool_input, context):  # noqa: ARG001
        from claude_agent_sdk.types import (
            PermissionResultAllow,
            PermissionResultDeny,
        )

        if not allow_writes and tool_name in WRITE_TOOLS:
            return PermissionResultDeny(
                message=f"tool {tool_name!r} is not allowed in this role",
            )

        if tool_name in PATH_FREE_TOOLS:
            return PermissionResultAllow(updated_input=tool_input)

        if tool_name not in PATH_KEYS:
            # Other tools (Bash, WebFetch, etc.) — deny by default. The
            # whole point of this guard is to keep the role inside the
            # library. If a future role legitimately needs another tool,
            # add it to PATH_KEYS (with appropriate path validation) or
            # to an explicit safe-list.
            return PermissionResultDeny(
                message=f"tool {tool_name!r} is not in the allow-list for this role",
            )

        for key in PATH_KEYS[tool_name]:
            raw = tool_input.get(key)
            if raw is None:
                # Optional path key (e.g. Glob without an explicit path
                # defaults to cwd, which we've already set inside root).
                continue
            try:
                resolved = Path(str(raw)).expanduser().resolve()
                resolved.relative_to(root)
            except (ValueError, OSError):
                return PermissionResultDeny(
                    message=(
                        f"path {raw!r} is outside the knowledge dir "
                        f"({root}); the {tool_name} call has been denied. "
                        f"Use a path under {root} instead."
                    ),
                )

        # Wikilink check on writes (Scholar only). Parses the proposed
        # content, auto-repairs links whose target moved to a new unique
        # path, and denies writes that introduce unresolvable links —
        # closes the "Scholar references an old path post-move" gap that
        # the post-write validator only logged about.
        if check_wikilinks and tool_name in WRITE_TOOLS:
            status, payload, info = _check_and_repair_writes(
                tool_name,
                tool_input,
                root,
            )
            if status == "deny":
                return PermissionResultDeny(message=info)
            if status == "allow_with_repair":
                log.warning(
                    "path guard: auto-repaired broken wikilinks in %s: %s",
                    tool_input.get("file_path"),
                    info,
                )
                return PermissionResultAllow(updated_input=payload)
            # status == "allow" — fall through to the default Allow.

        return PermissionResultAllow(updated_input=tool_input)

    return _can_use_tool


def _recursion_guard_env() -> dict:
    """Marker env vars passed into every SDK call.

    The hook layer is meant to check ``CLAUDE_INVOKED_BY`` and skip its
    work if set. Matches `src/scripts/flush.py`. We also inherit our own
    environment, which is what `asyncio.create_subprocess_*` does by
    default — but the SDK takes an explicit ``env=`` dict, so we pass it.
    """
    env = dict(os.environ)
    env["CLAUDE_INVOKED_BY"] = "agent_mem_daemon"
    return env


async def _drain_query(
    prompt: str,
    options,
) -> Tuple[str, float]:
    """Run one ``query(...)`` and return (full_text, cost_usd).

    The prompt is always passed as a streaming AsyncIterable rather
    than a bare string because the SDK requires streaming mode whenever
    a ``can_use_tool`` callback is set (see SDK ``client.py``: bare
    strings raise ``ValueError`` once a permission callback is wired
    up). Streaming with a single yielded message is functionally
    identical to one-shot mode but unlocks the permission-guard path.
    """
    # Lazy import so a missing SDK doesn't kill the daemon on startup —
    # only the actual Librarian/Scholar paths need it.
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )

    async def _prompt_stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": prompt},
        }

    full = ""
    cost = 0.0
    async for message in query(prompt=_prompt_stream(), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    full += block.text
                elif isinstance(block, ToolUseBlock):
                    # Inline a marker so the transcript shows what tools
                    # were called. Critical diagnostic — the Scholar's
                    # writes happen via tools, not in the text stream.
                    input_repr = str(block.input)[:300]
                    full += f"\n[tool: {block.name}({input_repr})]\n"
        elif isinstance(message, ResultMessage):
            # ResultMessage carries the cost the SDK reports for the run.
            cost = float(message.total_cost_usd or 0.0)
    return full, cost


def _run_with_timeout(coro, timeout_s: float):
    """Wrap a coroutine in ``asyncio.run`` with a timeout. Raises
    LLMTimeout on expiry."""

    async def _wrapped():
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise LLMTimeout(f"SDK call exceeded {timeout_s}s") from e

    return asyncio.run(_wrapped())


def run_librarian_call(
    prompt: str,
    *,
    cwd: Path | None = None,
    timeout_s: float = LIBRARIAN_TIMEOUT_S,
    model: str = LIBRARIAN_MODEL,
) -> Tuple[str, float]:
    """Run the Librarian Haiku call with read-only tools.

    The Librarian needs Read+Glob so it can inspect specific entries
    before proposing actions — the snapshot in the prompt is a teaser,
    not the full corpus. Tools are READ-ONLY: no Write, no Edit. The
    Scholar is the only writer.

    Args:
        prompt: full assembled prompt.
        cwd: working directory for the SDK — should be the knowledge
            store root so Read/Glob land in the right tree. Required when
            the Librarian is allowed to inspect files.
        timeout_s: SDK timeout.
        model: override the default Haiku model.

    Returns (response_text, cost_usd). Raises:
      - LLMTimeout if the call exceeded ``timeout_s``.
      - Any other exception the SDK raises (caller should catch and log).
    """
    # Lazy import so daemon startup doesn't require the SDK to be present.
    from claude_agent_sdk import ClaudeAgentOptions

    from . import library_tools

    librarian_tools = ["Read", "Glob", "Grep"]
    mcp_servers: dict = {}

    if cwd is not None:
        # Register the in-process BM25 search tool so the Librarian can
        # find semantically related entries without the daemon doing a
        # regex pre-pass. Complements Glob (filename) and Grep (literal).
        mcp_servers[library_tools._SERVER_NAME] = library_tools.make_library_mcp_server(cwd)
        librarian_tools.append(library_tools.fully_qualified_tool_name())

    opts: dict = dict(
        model=model,
        allowed_tools=librarian_tools,
        mcp_servers=mcp_servers,
        # Same budget as Scholar — observed traces show the Librarian
        # legitimately needs 30+ Read/Glob/bm25 calls on complex sessions
        # (multi-file dedup, cross-folder lookup) before emitting JSON;
        # capping lower was triggering "Reached maximum number of turns"
        # errors on real work.
        max_turns=100,
        env=_recursion_guard_env(),
        # The claude_code preset is what lets Read/Glob work end-to-end.
        # Without it the SDK exposes the tools but the model's harness
        # doesn't know how to act on results.
        system_prompt={"type": "preset", "preset": "claude_code"},
    )
    if cwd is not None:
        opts["cwd"] = str(cwd)
        # Infra-level path guard: deny any Read/Glob/Grep outside the
        # knowledge dir. Without this, the LLM can pass absolute paths
        # and bypass cwd. The bm25_search tool is path-safe by
        # construction (knowledge_dir captured in closure), so it
        # doesn't need guard coverage.
        opts["can_use_tool"] = _make_path_guard(cwd, allow_writes=False)
    options = ClaudeAgentOptions(**opts)
    log.debug(
        "librarian SDK call: model=%s cwd=%s prompt_chars=%d",
        model,
        cwd,
        len(prompt),
    )
    return _run_with_timeout(_drain_query(prompt, options), timeout_s)


def run_scholar_call(
    prompt: str,
    *,
    cwd: Path,
    timeout_s: float = SCHOLAR_TIMEOUT_S,
    model: str = SCHOLAR_MODEL,
) -> Tuple[str, float]:
    """Run the Scholar's file-tool-enabled Opus call.

    Args:
        prompt: full assembled prompt (template + packets).
        cwd: working directory for the SDK — should be the store dir
             (``~/.agent-mem/``), so Read/Glob/Grep land in the right
             tree. The Scholar writes via Write/Edit relative to this.

    Returns (response_text, cost_usd). Raises LLMTimeout on timeout; the
    caller logs and skips the batch.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    from . import library_tools

    # Scholar's cwd is the agent-mem home (so prompt-relative paths like
    # ``knowledge/index.md`` resolve correctly). The guard's boundary is
    # the narrower ``knowledge/`` subdir — Scholar must not touch
    # events.jsonl, daemon.log, runs/, cost.json, etc.
    boundary = cwd / "knowledge"
    # Register the in-process library MCP server so the Scholar can call
    # the atomic move_entries tool (which rewrites inbound wikilinks
    # programmatically — the LLM should NEVER move files by hand).
    mcp_servers = {
        library_tools._SERVER_NAME: library_tools.make_library_mcp_server(boundary),
    }
    options = ClaudeAgentOptions(
        model=model,
        cwd=str(cwd),
        system_prompt={"type": "preset", "preset": "claude_code"},
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            library_tools.fully_qualified_move_name(),
        ],
        mcp_servers=mcp_servers,
        permission_mode="acceptEdits",
        max_turns=100,
        env=_recursion_guard_env(),
        # Infra-level path guard: every Read/Write/Edit/Glob/Grep is
        # rejected if its path resolves outside the knowledge dir.
        # Stops the "Scholar edits the wrong store" class of bug at
        # the SDK layer, independent of whatever the prompt says.
        can_use_tool=_make_path_guard(
            boundary,
            allow_writes=True,
            check_wikilinks=True,
        ),
    )
    log.debug(
        "scholar SDK call: model=%s cwd=%s prompt_chars=%d",
        model,
        cwd,
        len(prompt),
    )
    return _run_with_timeout(_drain_query(prompt, options), timeout_s)
