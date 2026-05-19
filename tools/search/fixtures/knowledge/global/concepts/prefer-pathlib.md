---
id: prefer-pathlib
type: lesson
scope: global
status: confirmed
confidence: 0.8
applies-when: |
  touching filesystem paths in Python
keywords: [python, pathlib, os.path, filesystem]
created: 2026-05-19
updated: 2026-05-19
fired: 0
fired-helpful: 0
---

# Prefer pathlib over os.path

**Rule:** New Python code uses `pathlib.Path`, not `os.path` string manipulation.

**Why:** `Path` operations are explicit about what's a path vs a string, compose
cleanly with `/`, and have built-in `read_text`/`write_text` that close handles
correctly.

**How to apply:** `from pathlib import Path`. Convert legacy string paths at the
boundary, not throughout the codebase.
