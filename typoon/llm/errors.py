"""LLM error taxonomy used by stage workers to decide attempt accounting.

`OpenAIProvider` (and other adapters) classify upstream failures into
the categories below before reraising. The stage runner in
`typoon/workers/loop.py` distinguishes:

  - `TransientCredentialError` / `UpstreamUnavailable`
        ⇒ requeue WITHOUT incrementing attempts; sleep and retry on the
          next claim. Keeps a chapter alive across provider outages
          (token revoked, credential pool empty, gateway 5xx) so a
          config swap + worker restart resumes it instead of dead-
          lettering it.

  - any other `Exception`
        ⇒ count toward `MAX_TASK_ATTEMPTS` via `fail_task` (existing
          behavior). True bugs / data errors still dead-letter.

Adapters MUST raise these from inside `Provider.call` so the worker
sees them at the same boundary it already catches `Exception`.
"""

from __future__ import annotations


class TransientCredentialError(RuntimeError):
    """Upstream auth failure that the operator can fix without code change.

    Examples: 401 with `token_invalidated`, 403 with revoked key, proxy
    responses indicating the credential pool is empty (Bifrost emits
    503 `no available credential for provider`).
    """


class UpstreamUnavailable(RuntimeError):
    """Provider is reachable but cannot serve us right now.

    Examples: 502/503/504 from a gateway after the in-process retry
    budget is exhausted, model temporarily disabled, region routing
    outage. Treated like a credential error for attempt accounting —
    the chapter waits instead of burning attempts.
    """
