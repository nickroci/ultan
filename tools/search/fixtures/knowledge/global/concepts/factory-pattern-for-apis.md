---
id: factory-pattern-for-apis
type: lesson
scope: global
status: confirmed
confidence: 0.85
applies-when: |
  designing or building any new API
  decisions about how clients construct service objects
keywords: [factory, paradigm, api, construction]
created: 2026-05-19
updated: 2026-05-19
fired: 0
fired-helpful: 0
sources:
  - daily/2026-05-12.md#L42
related: [[connections/paradigms-cross-cutting]]
---

# Use factory pattern for new APIs

**Rule:** New service-facing APIs go through a factory, not direct constructor calls from
consumers.

**Why:** Factories let us swap implementations (test doubles, regional variants, feature
flags) without touching every call site. Direct constructors leak implementation details
into the consumer.

**How to apply:** Define a `*Factory.create()` returning the interface, not the concrete
class. Consumers depend on the factory, not the implementation.

```python
class PaymentsClientFactory:
    @staticmethod
    def create(region: str) -> PaymentsClient:
        ...
```
