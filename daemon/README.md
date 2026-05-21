# agent-mem-daemon

The long-running event-ingest daemon for **agent-mem** (see
`/Users/nicholasholden/agent-mem/PLAN.md`). This package is the
**skeleton phase**: it nails the plumbing — JSONL tail, rolling buffer,
turn aggregation, Librarian/Scholar scheduling, backpressure, PID/log
lifecycle — but the Librarian and Scholar themselves are stubs that log
"would have done X". The real LLM-driven Librarian/Scholar are a
separate deliverable.

## Install / run

```bash
cd /Users/nicholasholden/agent-mem/daemon
uv sync --group dev
uv run pytest                 # unit tests
uv run agent-mem-daemon -v    # foreground; logs to ~/.agent-mem/daemon.log + stderr
```

Defaults:

| Setting | Default | Flag |
|---|---|---|
| Events file | `~/.agent-mem/events.jsonl` | `--events-file` |
| Log file | `~/.agent-mem/daemon.log` | `--log-file` |
| PID file | `~/.agent-mem/daemon.pid` | `--pid-file` |
| Poll interval | 0.25s | `--poll-interval` |
| Rolling buffer turns / session | 20 | `--max-turns` |
| Session idle eviction | 3600s | `--inactivity-seconds` |
| Scholar every K Librarian runs | 3 | `--scholar-every-k` |
| Scholar every M seconds | 600 | `--scholar-every-m-secs` |
| Backpressure ceiling | 20 | `--queue-ceiling` |
| Session sweep cadence | 300s | `--sweep-interval-secs` |

The daemon refuses to start if `~/.agent-mem/daemon.pid` already names a
live process. Stale PID files (process gone) are silently overwritten.

## Frozen contracts

### 1. The event log — `~/.agent-mem/events.jsonl`

This is the contract between **the hooks** (a separate agent's
deliverable) and **the daemon**. The hooks append one JSON object per
line. The daemon tails the file.

Schema:

```json
{
  "ts": "2026-05-19T10:30:00.123Z",
  "session_id": "claude-session-abc-123",
  "type": "PostToolUse",
  "cwd": "/Users/me/code/my-project",
  "payload": { "tool": "Edit", "input": { "...": "..." }, "output": "..." }
}
```

Field-by-field:

| Field | Required | Type | Meaning |
|---|---|---|---|
| `ts` | yes | ISO-8601 string **or** numeric unix seconds | Event time. If missing the daemon falls back to receipt time and warns. |
| `session_id` | yes | string | Claude Code session identifier. Used as the rolling-buffer partition key. |
| `type` | yes | string | One of `PostToolUse`, `Stop`, `SessionEnd`, `UserPromptSubmit`, `SessionStart`, or any future custom string. Unknown types are accepted and logged at DEBUG. |
| `cwd` | no | string | Project working dir. Late-arriving `cwd` back-fills the session if the first event lacked it. |
| `payload` | no | object | Arbitrary. The daemon doesn't read inside it in this phase; the future Librarian will. |

Turn-aggregation rules the daemon assumes (PLAN §4):

- A **turn** ends at every `Stop` event.
- Everything between two `Stop` events belongs to the same turn,
  *including* the trailing `Stop` itself (so the future Librarian sees
  it as part of the turn).
- `SessionEnd` also seals the open turn and marks the session as ended.
- The hooks should write **one line per event**, terminated with `\n`.
  Partial lines are buffered until the trailing newline arrives.
- The hooks should **append**, never overwrite. The daemon detects
  rotation (inode change) and truncation (file size < last offset) and
  re-opens from the start in both cases.

Sessions idle for `--inactivity-seconds` (1h default) are evicted from
the in-memory buffer.

### 2. Stub callable signatures

The scheduler invokes two pluggable functions. The future LLM-backed
agent must keep these signatures stable:

```python
# agent_mem_daemon/librarian.py
def scan(buffer_snapshot: dict) -> EvidencePacket: ...

# agent_mem_daemon/scholar.py
def review(packets: list[EvidencePacket]) -> None: ...
```

`buffer_snapshot` is the output of `RollingBuffer.snapshot(session_id)`
and has shape:

```python
{
    "session_id": "...",
    "cwd": "..." | None,
    "ended": bool,
    "last_activity": float,
    "turns": [
        {
            "started_at": float,
            "sealed_at": float,
            "events": [
                {"ts": float, "type": str, "cwd": str|None, "payload": dict},
                ...
            ],
        },
        ...
    ],
}
```

`EvidencePacket` is intentionally a loose `TypedDict(total=False)`:

```python
class EvidencePacket(TypedDict, total=False):
    session_id: str
    candidates: list[dict]   # candidate lessons + evidence + dedup hits
    interrupts: list[dict]   # interrupt candidates + matching lesson refs
```

In the skeleton, both lists are always empty.

### 3. Scheduling invariants

- **Trigger.** Every `Stop` (or `SessionEnd`) flips
  `SessionState.needs_librarian` and invokes `librarian.scan` for that
  session.
- **Scholar.** Runs when **either** K Librarian invocations have
  accumulated **or** M seconds have elapsed since the last Scholar run,
  *whichever comes first*. K and M are configurable.
- **Backpressure.** When the Scholar's input queue length is
  `>= queue_ceiling` (default 20), new Librarian invocations are
  skipped (PLAN §4 final bullet). Skips are counted in stats and
  logged. The next time the Scholar drains the queue, Librarian
  invocations resume on the next `Stop`.
- **Shutdown drain.** On SIGTERM/SIGINT, the daemon calls
  `scheduler.force_scholar()` once before exit to flush whatever's
  queued.

## Dogfooding without the LLM half

You can exercise every code path right now without writing a hook or
calling an LLM. Open two terminals.

Terminal 1 (the daemon, foreground, verbose):

```bash
mkdir -p /tmp/agent-mem-test
AGENT_MEM_HOME=/tmp/agent-mem-test \
  uv run --project /Users/nicholasholden/agent-mem/daemon \
  agent-mem-daemon -v \
    --scholar-every-k 2 \
    --scholar-every-m-secs 30
```

Terminal 2 (a fake hook stream):

```bash
EVENTS=/tmp/agent-mem-test/events.jsonl
SID=$(uuidgen)
for i in 1 2 3; do
  printf '{"ts": %f, "session_id": "%s", "type": "PostToolUse", "cwd": "/tmp/proj", "payload": {"tool": "Edit", "i": %d}}\n' \
    "$(date +%s).0" "$SID" "$i" >> "$EVENTS"
  printf '{"ts": %f, "session_id": "%s", "type": "Stop", "cwd": "/tmp/proj"}\n' \
    "$(date +%s).0" "$SID" >> "$EVENTS"
  sleep 1
done
```

Terminal 1 will log, in order: three `STUB librarian.scan` lines (one
per `Stop`), then — once two of those have accumulated — one
`STUB scholar.review: would judge 2 packets` line. The third packet
sits in the queue until the next Scholar trigger (K=2 → next packet
fires it; M=30s → also fires it after the timer). Hit Ctrl-C and you
should see a final shutdown drain line.

To exercise rotation: `mv $EVENTS ${EVENTS}.1 && touch $EVENTS && echo
'{"ts":...,...}' >> $EVENTS` — the next poll-cycle log line will say
*"events file rotated (inode X -> Y); re-opening"*.

To exercise backpressure: lower `--queue-ceiling` to e.g. 2 and crank
the Scholar thresholds high (`--scholar-every-k 999
--scholar-every-m-secs 999`). After the third `Stop`, the daemon logs
*"BACKPRESSURE: scholar queue=2 >= ceiling=2; skipping librarian pass"*.

## Layout

```
daemon/
  pyproject.toml                       uv-managed; stdlib-only runtime
  agent_mem_daemon/
    __init__.py
    __main__.py                        entry, args, signals, PID file
    paths.py                           ~/.agent-mem/ resolution (AGENT_MEM_HOME-aware)
    logging_setup.py                   rotated file logging, cribbed from mann1x
    ingest.py                          JSONL tail (rotation/truncation/partial-line aware)
    buffer.py                          per-session rolling buffer, turn aggregation
    scheduler.py                       Librarian/Scholar cadence + backpressure
    librarian.py                       STUB — returns empty EvidencePacket
    scholar.py                         STUB — logs intent, writes nothing
  tests/
    test_buffer.py                     aggregation, eviction, snapshot shape
    test_ingest.py                     parse, append, rotation, truncation, partials
    test_scheduler.py                  cadence, backpressure, force-drain
  README.md                            this file
```

## TODOs left for later phases

- **Backgrounding.** v1 is foreground-only. Phase 4 will add a launchd
  plist on macOS. The `--foreground` flag is reserved so future
  invocations don't break.
- **LLM-backed Librarian / Scholar.** Replace `librarian.scan` and
  `scholar.review` (signatures stay).
- **Pending nudges file** (`~/.agent-mem/pending-nudges.md`). Path is
  reserved in `paths.py` for the future Scholar.
- **CLI** (`agent-mem review`, `search`, `doctor`). Out of scope here;
  lives in `src/`.
