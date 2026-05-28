"""Shared Pydantic-AI building blocks for the two curator agents.

The Librarian and the Scholar are both Pydantic AI ``Agent``s with the
SAME read-only research surface — read an entry, grep the library, BM25 +
embedding search — and the same run/cost plumbing. That code lives here,
ONCE, so the two agent modules stay focused on what differs (their
``output_type`` and their boundary ``output_validator``) and the
duplicate-code gate has nothing to flag.

What this module owns:

  - ``ResearchDeps`` — the dependency object every research tool reads
    (just the pinned ``knowledge_dir``). Both agents' deps types are this
    type, so the registered tools are deps-compatible across agents.
  - ``register_research_tools(agent)`` — wires the four read-only
    ``@agent.tool``s onto an agent built with ``deps_type=ResearchDeps``.
  - the pure tool helpers (``inside`` / ``grep_library`` / ``search_text``)
    so they can be unit-tested directly without a model in the loop.
  - ``estimate_cost`` (best-effort USD from token usage) and
    ``run_agent_to_output`` (the ``asyncio.run`` + wall-clock-timeout
    wrapper that both ``run_*_agent`` functions delegate to).

It imports only ``pydantic_ai`` + the daemon's ``library_tools`` (and,
lazily, ``genai_prices``) so either agent module can import it without a
cycle.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, TypeVar

from pydantic_ai import RunContext

from . import library_tools
from .llm import LLMTimeout

if TYPE_CHECKING:
    from pydantic_ai import Agent

log = logging.getLogger("agent_mem_daemon._agent_common")

_OutputT = TypeVar("_OutputT")


@dataclass
class ResearchDeps:
    """Dependencies handed to the shared read-only research tools.

    ``knowledge_dir`` is the pinned knowledge store root; the tools resolve
    every model-supplied path against it (no path injection). Each agent's
    own ``deps_type`` is this class (or a trivial subclass), so the tools
    registered by :func:`register_research_tools` work for both.
    """

    knowledge_dir: Path


# ── Pure tool helpers (module-level, easy to unit-test) ──────────────────


def inside(root: Path, candidate: Path) -> bool:
    """True iff ``candidate`` resolves inside ``root`` (no path escape)."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def grep_library(knowledge_dir: Path, pattern: str, path: str) -> str:
    """Case-insensitive literal-substring search over the library, optionally
    scoped to a subdirectory. Returns up to 40 ``<rel>:<line-no>: <line>``
    matches, a truncation note past that, or a sentinel. Read-only,
    ``_archive`` excluded."""
    root = knowledge_dir.resolve()
    if not pattern.strip():
        return "(grep_library: empty pattern)"
    scope = (root / path).resolve() if path else root
    if not inside(root, scope) or not scope.exists():
        return f"(grep_library: {path!r} not found under the knowledge store)"
    needle = pattern.lower()
    search_root = scope if scope.is_dir() else scope.parent
    out: List[str] = []
    for md in sorted(search_root.rglob("*.md")):
        if "_archive" in md.parts:
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                rel = md.relative_to(root).as_posix()
                out.append(f"{rel}:{i}: {line.strip()}")
                if len(out) >= 40:
                    return "\n".join(out) + "\n(truncated at 40 matches)"
    return "\n".join(out) if out else f"(no matches for {pattern!r})"


def search_text(
    runner: Callable[[Dict[str, Any], Path], Dict[str, Any]],
    knowledge_dir: Path,
    query: str,
    k: int,
) -> str:
    """Call a library search runner and unwrap its MCP-shaped response into
    the plain text the agent reads."""
    response = runner({"query": query, "k": k}, knowledge_dir.resolve())
    text = library_tools.unwrap_text_response(response)
    return text or "(search returned no content)"


# ── Read-only research tools (registered onto each agent) ────────────────


def register_research_tools(agent: "Agent[ResearchDeps, Any]") -> None:
    """Register the four read-only research ``@agent.tool``s onto ``agent``.

    ``agent`` must have been built with ``deps_type=ResearchDeps`` (or a
    subclass). The tools are READ-ONLY: there is no file-writing tool — the
    daemon's deterministic executor is the only writer. Called once at
    module import time by each agent module."""

    @agent.tool(name="read_entry")
    def read_entry(ctx: RunContext[ResearchDeps], path: str) -> str:
        """Read a knowledge file by its path relative to the knowledge root
        (e.g. ``index.md`` or ``global/python/use-uv.md``). Returns the file
        contents, or a ``(not found ...)`` sentinel. Read-only."""
        root = ctx.deps.knowledge_dir.resolve()
        target = (root / path).resolve()
        if not inside(root, target):
            return f"(path {path!r} resolves outside the knowledge store — refused)"
        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"(not found: {path})"
        except OSError as e:
            return f"(could not read {path}: {e})"

    @agent.tool(name="grep_library")
    def grep_library_tool(ctx: RunContext[ResearchDeps], pattern: str, path: str = "") -> str:
        """Search the knowledge library for a literal substring (case-
        insensitive), optionally scoped to a subdirectory ``path``. Returns
        up to 40 ``<rel-path>:<line-no>: <line>`` matches. Read-only."""
        return grep_library(ctx.deps.knowledge_dir, pattern, path)

    @agent.tool(name="bm25_search")
    def bm25_search(ctx: RunContext[ResearchDeps], query: str, k: int = 6) -> str:
        """Lexical (BM25) search over the library. Returns the top-K entries
        as ``<path>  score=<float>  <snippet>`` lines. Complements
        ``embedding_search``. Read-only."""
        return search_text(library_tools.run_bm25_search, ctx.deps.knowledge_dir, query, k)

    @agent.tool(name="embedding_search")
    def embedding_search(ctx: RunContext[ResearchDeps], query: str, k: int = 6) -> str:
        """Semantic (embedding) search over the library. Returns the top-K
        entries as ``<path>  score=<float>  <snippet>`` lines. Complements
        ``bm25_search`` — run both for any concept query. Read-only."""
        return search_text(library_tools.run_embedding_search, ctx.deps.knowledge_dir, query, k)

    # Reference the closures so linters don't flag them as unused — the
    # decorator already registered each on the agent.
    _ = (read_entry, grep_library_tool, bm25_search, embedding_search)


# ── Cost + run wrapper ───────────────────────────────────────────────────


def estimate_cost(model_ref: str, usage: object) -> float:
    """Best-effort USD cost for a run from token usage via ``genai_prices``.
    Returns 0.0 if pricing is unavailable — cost is telemetry, never on the
    critical path."""
    try:
        import genai_prices  # noqa: PLC0415 — optional, best-effort

        bare = model_ref.split(":", 1)[1] if ":" in model_ref else model_ref
        calc = genai_prices.calc_price(usage, bare, provider_id="anthropic")  # type: ignore[arg-type]
        return float(calc.total_price)
    except Exception:  # noqa: BLE001 — pricing must never break the pipeline
        log.debug("agent cost estimation unavailable", exc_info=True)
        return 0.0


def run_agent_to_output(
    agent: "Agent[ResearchDeps, _OutputT]",
    prompt: str,
    deps: ResearchDeps,
    *,
    model_ref: str,
    timeout_s: float,
    role: str,
) -> Tuple[_OutputT, float]:
    """Run ``agent`` on ``prompt`` and return ``(validated_output, cost_usd)``.

    Wraps the agent run in ``asyncio.run`` with a wall-clock budget; raises
    :class:`LLMTimeout` (which the daemon already handles) when exceeded, and
    propagates any other agent/model error to the caller. ``role`` only
    flavours the timeout message. Pydantic AI has already run the per-output
    validators and the agent's ``output_validator`` (re-prompting on any
    ``ModelRetry`` up to the agent's output-retry budget) before this
    returns."""

    async def _run() -> Tuple[_OutputT, float]:
        try:
            result = await asyncio.wait_for(
                agent.run(prompt, deps=deps, model_settings={"timeout": timeout_s}),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise LLMTimeout(f"{role} agent exceeded {timeout_s}s") from e
        return result.output, estimate_cost(model_ref, result.usage)

    return asyncio.run(_run())
