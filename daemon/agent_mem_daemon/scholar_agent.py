"""The Scholar as a Pydantic AI agent.

This replaces the old Claude-Agent-SDK ``run_scholar_call`` +
hand-scraped-JSON path. The Scholar is now a typed agent that:

  - runs on ``anthropic:claude-opus-4-7``,
  - returns a typed, validated ``ScholarDecisions`` (``output_type``),
  - has READ-ONLY verification tools (read an entry, grep the library,
    BM25 + embedding search) wired as direct in-process ``@agent.tool``s —
    it has NO file-writing tools; the daemon executor applies the actions,
  - validates its own output at the boundary via an ``@agent.output_validator``
    (context-dependent checks that need the knowledge dir + the whole
    batch), on top of the per-action Pydantic validators in ``_schemas``.

On any boundary-validation failure the validator raises ``ModelRetry`` with
a specific, actionable message so the model fixes the offending action and
re-emits. The output retry budget is set on the agent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Set, Tuple

from pydantic_ai import Agent, ModelRetry, RunContext

from . import _agent_common, _validation
from ._agent_common import ResearchDeps as ScholarDeps  # the Scholar's deps type
from ._agent_common import estimate_cost as _estimate_cost
from ._agent_common import grep_library as _grep_library
from ._agent_common import inside as _inside
from ._agent_common import search_text as _search_text
from ._schemas import ScholarDecisions

if TYPE_CHECKING:
    from ._schemas import ScholarAction

# The shared research helpers live in ``_agent_common`` now; re-export the
# names the Scholar's unit tests reach for so they keep working without
# knowing the helpers moved. Listed here (not just imported) so pyright
# treats them as the module's intended public surface, not dead imports.
__all__ = [
    "SCHOLAR_AGENT",
    "SCHOLAR_MODEL",
    "SCHOLAR_TIMEOUT_S",
    "ScholarDeps",
    "run_scholar_agent",
    "validate_decisions",
    "_estimate_cost",
    "_grep_library",
    "_inside",
    "_search_text",
]

log = logging.getLogger("agent_mem_daemon.scholar_agent")

SCHOLAR_MODEL = "anthropic:claude-opus-4-7"

# Wall-clock budget for one Scholar agent run (matches the old SDK timeout).
SCHOLAR_TIMEOUT_S = 600.0

# Output-validation retry budget. Each ModelRetry from the validator (or a
# per-action Pydantic ValidationError) consumes one; after this many the run
# raises and the daemon drops the batch (the lessons recur next session).
OUTPUT_RETRIES = 4


# The Scholar agent, built once at module import. ``defer_model_check=True``
# lets the daemon import this module without an ``ANTHROPIC_API_KEY`` present
# — the Anthropic provider is only instantiated when ``run`` actually fires.
# The read-only research tools are shared with the Librarian and registered
# via ``_agent_common``; the output validator below is Scholar-specific.
SCHOLAR_AGENT: "Agent[ScholarDeps, ScholarDecisions]" = Agent(
    SCHOLAR_MODEL,
    output_type=ScholarDecisions,
    deps_type=ScholarDeps,
    retries={"output": OUTPUT_RETRIES},
    defer_model_check=True,
)

_agent_common.register_research_tools(SCHOLAR_AGENT)


# ── Context-dependent boundary validation ────────────────────────────────


@SCHOLAR_AGENT.output_validator
def validate_decisions(ctx: RunContext[ScholarDeps], output: ScholarDecisions) -> ScholarDecisions:
    """Whole-batch checks that the per-action Pydantic validators can't do
    in isolation. Raises ``ModelRetry`` with a specific message on the first
    failure so the model fixes it and re-emits."""
    root = ctx.deps.knowledge_dir.resolve()
    _validate_wikilinks(output, root)
    _validate_flat_dir_caps(output, root)
    return output


# ── Output-validator helpers ─────────────────────────────────────────────


def _action_body_and_path(action: "ScholarAction") -> Tuple[str, str]:
    """Return ``(body, path)`` for the body-carrying actions, or
    ``("", path)`` / ``("", "")`` for the rest."""
    kind = str(action.action)
    if kind == "write_entry":
        return getattr(action, "body", ""), getattr(action, "path", "")
    if kind == "update_entry":
        return getattr(action, "new_body", ""), getattr(action, "path", "")
    if kind == "merge_entries":
        return getattr(action, "target_body", ""), getattr(action, "target_path", "")
    return "", ""


def _created_paths(output: "ScholarDecisions") -> Set[str]:
    """The set of entry wikilink targets (no ``.md``) that this batch will
    create or relocate-to — so a body may legitimately link to a sibling
    action's output."""
    created: Set[str] = set()
    for action in output.actions:
        kind = str(action.action)
        if kind in ("write_entry", "update_entry"):
            created.add(_strip_md(getattr(action, "path", "")))
        elif kind == "merge_entries":
            created.add(_strip_md(getattr(action, "target_path", "")))
        elif kind == "move_entry":
            created.add(_strip_md(getattr(action, "to_path", "")))
    created.discard("")
    return created


def _strip_md(path: str) -> str:
    return path[:-3] if path.endswith(".md") else path


def _validate_wikilinks(output: "ScholarDecisions", root: Path) -> None:
    """Every ``[[wikilink]]`` in a returned body must resolve to an existing
    entry OR to a path another action in the same batch creates. Raises
    ``ModelRetry`` naming the first offending link."""
    created = _created_paths(output)
    for action in output.actions:
        body, path = _action_body_and_path(action)
        if not body:
            continue
        parent = (root / path).parent
        for link in _validation.body_wikilinks(body):
            if _strip_md(link) in created:
                continue
            if _validation.wikilink_resolves(link, parent, root):
                continue
            raise ModelRetry(
                f"action {action.action} for {path!r} contains an unresolvable "
                f"wikilink [[{link}]] — it points at neither an existing entry "
                f"nor a path created by another action in this batch. Fix the "
                f"target (full path from the knowledge root, no .md), add the "
                f"action that creates it, or remove the link, then re-emit."
            )


def _current_dir_counts(root: Path) -> Dict[str, int]:
    """Count entry .md files (excluding README/index/log and _archive) per
    directory, keyed by the directory's knowledge-relative posix path."""
    counts: Dict[str, int] = {}
    if not root.exists():
        return counts
    for md in root.rglob("*.md"):
        if "_archive" in md.parts:
            continue
        if md.name in ("README.md", "index.md", "log.md"):
            continue
        rel_dir = md.parent.relative_to(root).as_posix()
        counts[rel_dir] = counts.get(rel_dir, 0) + 1
    return counts


def _dir_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _apply_count_deltas(counts: Dict[str, int], output: "ScholarDecisions") -> None:
    """Mutate ``counts`` with the per-directory deltas this batch would cause
    (new writes add to their dir; moves shift between dirs; merges add the
    target and drop archived sources)."""
    for action in output.actions:
        kind = str(action.action)
        if kind == "write_entry":
            counts[_dir_of(getattr(action, "path", ""))] = (
                counts.get(_dir_of(getattr(action, "path", "")), 0) + 1
            )
        elif kind == "move_entry":
            counts[_dir_of(getattr(action, "to_path", ""))] = (
                counts.get(_dir_of(getattr(action, "to_path", "")), 0) + 1
            )
            src_dir = _dir_of(getattr(action, "from_path", ""))
            counts[src_dir] = max(0, counts.get(src_dir, 0) - 1)
        elif kind == "merge_entries":
            counts[_dir_of(getattr(action, "target_path", ""))] = (
                counts.get(_dir_of(getattr(action, "target_path", "")), 0) + 1
            )
            for src in getattr(action, "source_paths", []):
                if src == getattr(action, "target_path", ""):
                    continue
                d = _dir_of(src)
                counts[d] = max(0, counts.get(d, 0) - 1)
        elif kind == "archive_entry":
            d = _dir_of(getattr(action, "path", ""))
            counts[d] = max(0, counts.get(d, 0) - 1)


def _validate_flat_dir_caps(output: "ScholarDecisions", root: Path) -> None:
    """No directory may exceed ``MAX_FLAT_DIR_ENTRIES`` after this batch is
    applied. Raises ``ModelRetry`` naming the first over-cap directory."""
    counts = _current_dir_counts(root)
    _apply_count_deltas(counts, output)
    for rel_dir, n in sorted(counts.items()):
        if n > _validation.MAX_FLAT_DIR_ENTRIES:
            raise ModelRetry(
                f"applying these actions would leave {n} entries in "
                f"{rel_dir}/ (cap is {_validation.MAX_FLAT_DIR_ENTRIES}). Add "
                f"move_entry action(s) to rebalance into a subfolder, or drop "
                f"the write that pushes it over, then re-emit."
            )


def run_scholar_agent(
    prompt: str,
    knowledge_dir: Path,
    *,
    timeout_s: float = SCHOLAR_TIMEOUT_S,
) -> Tuple["ScholarDecisions", float]:
    """Run the Scholar agent on ``prompt`` and return
    ``(validated_decisions, cost_usd)``.

    The model emits a typed ``ScholarDecisions``; Pydantic AI runs the
    per-action validators and the ``output_validator`` (re-prompting the
    model on ``ModelRetry`` up to the output retry budget) before returning.
    Raises :class:`LLMTimeout` if the wall-clock budget is exceeded, and
    propagates any other agent/model error for the caller to log.
    """
    deps = ScholarDeps(knowledge_dir=knowledge_dir.resolve())
    return _agent_common.run_agent_to_output(
        SCHOLAR_AGENT,
        prompt,
        deps,
        model_ref=SCHOLAR_MODEL,
        timeout_s=timeout_s,
        role="Scholar",
    )
