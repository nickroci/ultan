"""Tests for the transcript-tail priming-query backfill (ultan/_transcript_tail).

The module's contract has two halves:
  1. The GATE — only low-signal prompts get backfilled. Rich prompts must
     pass through verbatim (the 2026-07-03 sweep showed context appended to
     a rich prompt silences good firings).
  2. The EXTRACTION — only conversational prose reaches the query: user
     prompt strings and assistant text blocks. Tool traffic, thinking,
     sidechains and system-reminder spans (prior priming!) must never
     feed back into retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultan import _transcript_tail as tt

# ── is_low_signal (the gate) ──────────────────────────────────────────


def test_confirmation_prompts_are_low_signal() -> None:
    for prompt in ("ok", "yes", "go ahead", "continue", "sounds good, do it", ""):
        assert tt.is_low_signal(prompt), prompt


def test_rich_prompts_are_not_low_signal() -> None:
    for prompt in (
        "check why we see no collateral timings for ZOREP",
        "not separate postgres per client, we should aim for row level permissioning",
        "refactor the priming rpc handler to batch surface bookkeeping",
    ):
        assert not tt.is_low_signal(prompt), prompt


def test_short_but_contentful_prompt_still_gates_in() -> None:
    # 2 content tokens survive the stopword pass (meant, pager) — under
    # the threshold, so the tail is appended. This is the real "one pager"
    # miss from the feedback log: the deciding signal lived in the
    # previous turn.
    assert tt.is_low_signal("this is meant to be a one pager")


# ── transcript extraction ─────────────────────────────────────────────


def _write_transcript(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")


def _user(text: str, **extra: Any) -> dict[str, Any]:
    return {"type": "user", "message": {"role": "user", "content": text}, **extra}


def _assistant(blocks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"type": "assistant", "message": {"role": "assistant", "content": blocks}, **extra}


def test_tail_extracts_user_strings_and_assistant_text_blocks(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    _write_transcript(
        f,
        [
            _user("we need to parse the trade economics"),
            _assistant(
                [
                    {"type": "thinking", "thinking": "SECRET internal reasoning"},
                    {"type": "text", "text": "I'll parse the JSON by hand with regex."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "rm -rf /tmp/x"}},
                ]
            ),
            # Tool result traffic arrives as a user-type entry with list content.
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "TOOL OUTPUT NOISE"}],
                },
            },
        ],
    )
    tail = tt.transcript_tail(f)
    assert "trade economics" in tail
    assert "parse the JSON by hand" in tail
    assert "SECRET" not in tail
    assert "TOOL OUTPUT NOISE" not in tail
    assert "rm -rf" not in tail


def test_tail_skips_sidechain_and_strips_system_reminders(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    _write_transcript(
        f,
        [
            _user("real prompt about docker versions"),
            _assistant([{"type": "text", "text": "subagent chatter"}], isSidechain=True),
            _user(
                "another turn <system-reminder>## Ultan — your library says\n"
                "- [[projects/x/y]] echo bait</system-reminder> tail text"
            ),
        ],
    )
    tail = tt.transcript_tail(f)
    assert "docker versions" in tail
    assert "subagent chatter" not in tail
    assert "echo bait" not in tail  # prior priming must not feed back in
    assert "tail text" in tail


def test_tail_takes_last_max_chars_and_survives_garbage(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    lines = ["not json at all", json.dumps(["a", "list"])]
    lines += [json.dumps(_user(f"turn {i} " + "filler words " * 10)) for i in range(50)]
    f.write_text("\n".join(lines), encoding="utf-8")
    tail = tt.transcript_tail(f, max_chars=200)
    assert len(tail) <= 200
    assert "turn 49" in tail  # most recent content wins


def test_tail_missing_file_and_nonpositive_budget_return_empty(tmp_path: Path) -> None:
    assert tt.transcript_tail(tmp_path / "absent.jsonl") == ""
    f = tmp_path / "t.jsonl"
    _write_transcript(f, [_user("hello world content")])
    assert tt.transcript_tail(f, max_chars=0) == ""


def test_tail_reads_only_the_end_of_huge_transcripts(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    junk = json.dumps(_user("ancient history " * 50))
    payload = [junk] * 700  # ~500 KB, well past the read-back window
    payload.append(json.dumps(_user("the recent decision about hyperthrusters")))
    f.write_text("\n".join(payload), encoding="utf-8")
    tail = tt.transcript_tail(f)
    assert "hyperthrusters" in tail


# ── build_priming_query (gate + extraction combined) ──────────────────


def test_rich_prompt_goes_verbatim_even_with_transcript(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    _write_transcript(f, [_assistant([{"type": "text", "text": "unrelated context"}])])
    rich = "check why we see no collateral timings for ZOREP"
    assert tt.build_priming_query(rich, str(f)) == rich


def test_low_signal_prompt_gets_tail_with_prompt_last(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    _write_transcript(
        f,
        [
            _assistant(
                [
                    {
                        "type": "text",
                        "text": "We must use spaceship B with hyperthrusters to fly to Jupiter.",
                    }
                ]
            )
        ],
    )
    q = tt.build_priming_query("ok", str(f))
    assert q.endswith("\nok")
    assert "hyperthrusters" in q


def test_low_signal_prompt_degrades_verbatim_without_transcript(tmp_path: Path) -> None:
    assert tt.build_priming_query("ok", None) == "ok"
    assert tt.build_priming_query("ok", str(tmp_path / "absent.jsonl")) == "ok"
