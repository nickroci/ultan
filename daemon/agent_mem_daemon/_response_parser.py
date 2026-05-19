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

    stripped = text.strip()

    # ── Step 1: strip a markdown fence if present ────────────────────
    if stripped.startswith("```"):
        # Drop the opening fence line (```json or ``` or ```anything).
        lines = stripped.splitlines()
        inner = lines[1:]
        # Drop trailing closing fence if present.
        if inner and inner[-1].lstrip().startswith("```"):
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()

    # ── Step 2: collect every top-level balanced { ... } ─────────────
    blocks: list[str] = []
    depth = 0
    in_string = False
    escape = False
    start = -1

    for i, ch in enumerate(stripped):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(stripped[start : i + 1])
                start = -1

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
