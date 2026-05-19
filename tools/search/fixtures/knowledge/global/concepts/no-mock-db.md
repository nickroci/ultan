---
id: no-mock-db
type: lesson
scope: global
status: confirmed
confidence: 0.9
applies-when: |
  writing or reviewing database tests
  decisions about how to test data-access code
keywords: [postgres, database, tests, mocking, integration]
created: 2026-05-19
updated: 2026-05-19
fired: 0
fired-helpful: 0
sources:
  - daily/2026-05-10.md#L88
---

# Don't mock the database

**Rule:** Tests that touch the database run against a real Postgres instance (testcontainers
or a local Docker container), never against an in-memory fake or a mock.

**Why:** SQL semantics, transactions, constraints, and index behavior are exactly the
things tests should be catching. Mocked databases let bugs through that production
won't.

**How to apply:** Use `testcontainers-python` (or equivalent) to spin Postgres per test
session. Apply migrations as part of fixture setup.
