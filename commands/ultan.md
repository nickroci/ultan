---
description: Save a memory to Ultan (cross-session agent memory). Usage /ultan <text>
argument-hint: <the memory text to remember>
allowed-tools: Bash(ultan remember:*)
disable-model-invocation: true
---

The user wants to commit something to long-term Ultan memory. Run it now:

!`ultan remember "$ARGUMENTS"`

The command above appends a user-asserted event to the daemon's queue; the
Librarian picks it up and files it. Report the confirmation line ("queued for
librarian …") back to the user in one short sentence. Do not do anything else.
