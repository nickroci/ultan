"""Shared JSON-blob extraction + Pydantic validation for LLM responses.

Both the Librarian and the Scholar wrap their structured output in
prose / code fences / trailing commentary at varying rates. This module
centralises:

1. ``extract_json_blob`` — find the largest balanced ``{...}`` block in
   the response, stripping markdown code fences and prose noise.
2. ``parse_response`` — extract the blob, try ``json.loads``, fall back
   to ``json_repair.loads`` on failure, then validate against a
   Pydantic model. Returns the model instance plus a small diagnostic
   record so callers can log what happened.

The fall-back chain matters: ``json_repair`` handles trailing commas,
unquoted keys, smart-quote substitution, and other Haiku/Opus quirks
without us having to anticipate them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Type, TypeVar

from json_repair import repair_json
from pydantic import BaseModel, ValidationError

log = logging.getLogger("agent_mem_daemon._response_parser")


T = TypeVar("T", bound=BaseModel)


@dataclass
class ParseDiagnostic:
    """Why a parse succeeded or failed, for logging.

    Always populated; callers can read it regardless of outcome.
    """

    ok: bool = False
    repair_applied: bool = False
    raw_json: Optional[str] = None
    error: Optional[str] = field(default=None)


def _strip_code_fence(text: str) -> str:
    """Peel a leading markdown code fence (```json ... ```) if present."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    inner = lines[1:]
    if inner and inner[-1].lstrip().startswith("```"):
        inner = inner[:-1]
    return "\n".join(inner).strip()


@dataclass
class _BraceScanner:
    """Mutable state for the balanced-brace walk. Separating it from
    ``_collect_balanced_blocks`` keeps the loop body straight-line."""

    depth: int = 0
    in_string: bool = False
    escape: bool = False
    start: int = -1


def _advance_scanner(state: _BraceScanner, i: int, ch: str) -> Optional[Tuple[int, int]]:
    """Feed one character into the scanner. Returns a ``(start, end)``
    slice (inclusive end) when a top-level balanced block just closed,
    else ``None``."""
    if state.escape:
        state.escape = False
        return None
    if state.in_string:
        if ch == "\\":
            state.escape = True
        elif ch == '"':
            state.in_string = False
        return None
    if ch == '"':
        state.in_string = True
        return None
    if ch == "{":
        if state.depth == 0:
            state.start = i
        state.depth += 1
        return None
    if ch == "}":
        state.depth -= 1
        if state.depth == 0 and state.start >= 0:
            block = (state.start, i)
            state.start = -1
            return block
    return None


def _collect_balanced_blocks(text: str) -> list[str]:
    """Walk ``text`` left-to-right and return every top-level balanced
    ``{...}`` substring. String-aware so braces inside JSON strings don't
    confuse the depth counter."""
    blocks: list[str] = []
    state = _BraceScanner()
    for i, ch in enumerate(text):
        closed = _advance_scanner(state, i, ch)
        if closed is not None:
            s, e = closed
            blocks.append(text[s : e + 1])
    return blocks


def extract_json_blob(text: str) -> Optional[str]:
    """Find the LAST top-level balanced ``{...}`` block in ``text``.

    We prefer the LAST block (not the first) because LLM responses
    typically end with the structured output, preceded by tool-call
    markers (``[tool: foo({'arg': 'val'})]``) whose Python-repr dicts
    contain ``{`` characters that look like JSON to a first-occurrence
    scanner.

    Strategy:
      1. If the input is wrapped in a markdown code fence
         (```json ... ``` or just ``` ... ```), peel it.
      2. Walk the string left-to-right, collecting every top-level
         balanced ``{...}`` substring (depth counter, string-aware).
      3. Return the LAST one. If none is balanced, fall back to the
         tail of the last unmatched ``{`` so json_repair gets a chance.
      4. Return ``None`` if no ``{`` was seen at all.
    """
    if not text or not text.strip():
        return None

    stripped = _strip_code_fence(text.strip())
    blocks = _collect_balanced_blocks(stripped)
    if blocks:
        return blocks[-1]

    # No balanced blocks — fall back to the tail from the last unmatched
    # ``{`` so json_repair has something to chew on.
    last_open = stripped.rfind("{")
    if last_open < 0:
        return None
    return stripped[last_open:]


def parse_response(
    text: str,
    model: Type[T],
) -> Tuple[Optional[T], ParseDiagnostic]:
    """Extract a JSON blob from ``text`` and validate against ``model``.

    Returns ``(instance_or_None, diagnostic)``. ``diagnostic.ok`` mirrors
    "we got a validated instance back". ``diagnostic.repair_applied`` is
    True iff we had to fall back to ``json_repair``.
    """
    diag = ParseDiagnostic()
    if not text or not text.strip():
        diag.error = "empty response"
        return None, diag

    blob = extract_json_blob(text)
    if blob is None:
        diag.error = "no JSON blob found in response"
        return None, diag
    diag.raw_json = blob

    obj = None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        # Fall back to json-repair, which tolerates trailing commas,
        # unquoted keys, smart quotes, and the other small-model JSON
        # quirks we routinely see from Haiku.
        try:
            repaired = repair_json(blob)
            obj = json.loads(repaired)
            diag.repair_applied = True
            diag.raw_json = repaired
        except (json.JSONDecodeError, ValueError) as e2:
            diag.error = f"json.loads failed ({e}); repair also failed ({e2})"
            return None, diag

    if not isinstance(obj, dict):
        diag.error = f"top-level JSON is not an object: {type(obj).__name__}"
        return None, diag

    try:
        validated = model.model_validate(obj)
    except ValidationError as e:
        diag.error = f"pydantic validation failed: {e}"
        return None, diag

    diag.ok = True
    return validated, diag
