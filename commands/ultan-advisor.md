---
description: Consult Ultan's stored preferences before asking the user a preference-shaped question. Usage /ultan-advisor <the question you were going to ask>
argument-hint: <the question or taste/convention decision>
allowed-tools: Bash(ultan advisor:*)
disable-model-invocation: true
---

Consult the user's Ultan memory library before asking them this question — they
may have already answered it in a previous session. Run:

```
ultan advisor "$ARGUMENTS"
```

This runs the two-step advisor pipeline (Librarian finds relevant entries, then
the Scholar writes a referenced markdown answer with `[[wikilink]]` citations).
It can take 30-60s and is read-only — it never writes to the library. Use a
generous Bash timeout (at least 120s).

When it returns:
- If the library has a relevant stored preference, follow it and tell the user
  what Ultan already remembers, citing the `[[wikilink]]`. Don't re-ask what
  they've already answered.
- If the library is silent on the question, say so plainly, then proceed and ask
  the user directly.
