---
id: paradigms-cross-cutting
type: reference
scope: global
status: confirmed
confidence: 0.7
applies-when: |
  discussions about overall code style or paradigms
keywords: [paradigm, style, architecture, conventions]
created: 2026-05-19
updated: 2026-05-19
---

# Cross-cutting paradigms

Collected paradigms applied across projects.

- Factories over constructors for service-facing APIs — see [[concepts/factory-pattern-for-apis]].
- Real databases in tests — see [[concepts/no-mock-db]].
- `pathlib.Path` over `os.path` in Python — see [[concepts/prefer-pathlib]].

These aren't religious — they're the defaults. Deviate with a comment explaining why.
