---
id: auth-redirects
type: gotcha
scope: project:example-app
status: provisional
confidence: 0.6
applies-when: |
  handling authentication redirects
  Supabase auth callback flows
keywords: [auth, redirect, supabase, callback, query-string]
created: 2026-05-19
updated: 2026-05-19
---

# Auth redirects must preserve query string

**Rule:** When redirecting through the auth callback, preserve the original `?next=…` query
string so the user lands on the page they were trying to reach.

**Why:** Otherwise users get bounced to the dashboard after login regardless of where they
clicked from. Annoying, and confuses analytics.

**How to apply:** In the `/auth/callback` handler, read `next` from the URL and pass it
through `supabase.auth.exchangeCodeForSession`'s redirect.
