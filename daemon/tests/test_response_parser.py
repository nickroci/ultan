"""Tests for the shared JSON extractor + Pydantic-validating parser.

Covers:
  - clean JSON
  - JSON wrapped in ```json ... ``` code fences
  - JSON with trailing prose / preamble
  - malformed JSON that json-repair can fix
  - unrecoverable garbage
  - balanced-brace extraction (braces inside string literals shouldn't
    confuse the depth counter)
"""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field, ConfigDict

from agent_mem_daemon._response_parser import (
    ParseDiagnostic,
    extract_json_blob,
    parse_response,
)


class _ToyModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    items: list[str] = Field(default_factory=list)


# ── extract_json_blob ────────────────────────────────────────────────


def test_extract_returns_clean_object_unchanged():
    src = '{"a": 1}'
    assert extract_json_blob(src) == '{"a": 1}'


def test_extract_strips_json_code_fence():
    src = '```json\n{"a": 1}\n```'
    assert extract_json_blob(src) == '{"a": 1}'


def test_extract_strips_bare_code_fence():
    src = '```\n{"a": 1}\n```'
    assert extract_json_blob(src) == '{"a": 1}'


def test_extract_skips_leading_prose():
    src = 'Sure, here you go:\n\n{"a": 1, "b": 2}'
    assert extract_json_blob(src) == '{"a": 1, "b": 2}'


def test_extract_stops_at_balanced_brace_not_trailing_prose():
    src = '{"a": 1}\n\nHope that helps!'
    assert extract_json_blob(src) == '{"a": 1}'


def test_extract_handles_braces_inside_strings():
    # The closing } inside the string must NOT terminate the object.
    src = '{"msg": "this has } a brace inside"}'
    out = extract_json_blob(src)
    assert out is not None
    # Round-trip parse must work.
    assert json.loads(out)["msg"] == "this has } a brace inside"


def test_extract_handles_escaped_quotes_in_strings():
    src = '{"msg": "she said \\"hi\\" and left"}'
    out = extract_json_blob(src)
    assert out is not None
    assert json.loads(out)["msg"] == 'she said "hi" and left'


def test_extract_nested_objects():
    src = '{"outer": {"inner": {"k": "v"}}}'
    out = extract_json_blob(src)
    assert out is not None
    assert json.loads(out)["outer"]["inner"]["k"] == "v"


def test_extract_returns_none_when_no_brace():
    assert extract_json_blob("just some words") is None
    assert extract_json_blob("") is None
    assert extract_json_blob("   ") is None


def test_extract_returns_partial_blob_when_unbalanced():
    # Unbalanced — we hand the rest of the string to json_repair upstream.
    src = '{"a": 1'
    out = extract_json_blob(src)
    assert out == '{"a": 1'


# ── parse_response (toy model) ───────────────────────────────────────


def test_parse_clean_json():
    obj, diag = parse_response('{"name": "x", "items": ["a", "b"]}', _ToyModel)
    assert isinstance(diag, ParseDiagnostic)
    assert diag.ok is True
    assert diag.repair_applied is False
    assert obj is not None
    assert obj.name == "x"
    assert obj.items == ["a", "b"]


def test_parse_fenced_json():
    src = 'Here is the JSON:\n```json\n{"name": "y"}\n```'
    obj, diag = parse_response(src, _ToyModel)
    assert diag.ok is True
    assert obj is not None
    assert obj.name == "y"


def test_parse_trailing_prose_after_object():
    src = '{"name": "z"}\nThanks!'
    obj, diag = parse_response(src, _ToyModel)
    assert diag.ok is True
    assert obj is not None
    assert obj.name == "z"


def test_parse_malformed_recoverable_via_json_repair():
    # Trailing comma — json.loads fails, json_repair fixes it.
    src = '{"name": "abc", "items": ["a", "b",]}'
    obj, diag = parse_response(src, _ToyModel)
    assert diag.ok is True
    assert diag.repair_applied is True
    assert obj is not None
    assert obj.name == "abc"
    assert obj.items == ["a", "b"]


def test_parse_unquoted_keys_recoverable():
    src = "{name: 'abc'}"
    obj, diag = parse_response(src, _ToyModel)
    # json_repair handles single quotes + unquoted keys.
    assert diag.ok is True
    assert diag.repair_applied is True
    assert obj is not None
    assert obj.name == "abc"


def test_parse_extra_fields_are_ignored():
    src = '{"name": "x", "items": [], "bonus": 42, "nested": {"x": 1}}'
    obj, diag = parse_response(src, _ToyModel)
    assert diag.ok is True
    assert obj is not None
    assert obj.name == "x"


def test_parse_completely_unrecoverable_garbage():
    obj, diag = parse_response("not even close to json", _ToyModel)
    assert diag.ok is False
    assert obj is None
    assert diag.error is not None


def test_parse_empty_input():
    obj, diag = parse_response("", _ToyModel)
    assert obj is None
    assert diag.ok is False
    assert diag.error == "empty response"


def test_parse_top_level_array_rejected():
    # We require an object at the top level.
    obj, diag = parse_response("[1, 2, 3]", _ToyModel)
    assert obj is None
    assert diag.ok is False


def test_parse_diagnostic_carries_repaired_blob():
    src = '{"name": "abc", "items": ["a", "b",]}'
    _obj, diag = parse_response(src, _ToyModel)
    assert diag.raw_json is not None
    # After repair the blob should be valid JSON.
    json.loads(diag.raw_json)
