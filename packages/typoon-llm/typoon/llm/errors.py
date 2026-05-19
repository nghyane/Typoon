"""LLM error taxonomy used by stage workers to decide attempt accounting.

Three layers, matching what an operator can actually do about each:

  - `OperatorActionRequired`
        The provider rejected the request in a way no amount of
        retrying will fix without human intervention: model not
        present in the routing group, credential pool empty, billing
        suspended, region disabled. Raising this PAUSES the whole
        stage in `stage_pause` — workers stop claiming `translate`
        (or whichever stage raised) until an admin resumes via the
        ops CLI. The task itself stays at `attempts=0`, so once the
        cause is fixed and the stage resumed, the same chapter
        continues from where it left off.

  - `UpstreamUnavailable`
        Provider is reachable but is currently failing in a way that
        usually heals by itself: 502/504 from a gateway after the
        in-process retry budget is exhausted, transient timeouts,
        rate limits past their bucket. The task is released without
        bumping `attempts` and re-claimed after an exponential
        backoff. Long enough outages eventually drift into operator
        territory (manual escalation), but the worker won't burn
        cycles in the meantime.

  - any other `Exception`
        Treated as a programming or data bug — `fail_task` sets
        `attempts = MAX_TASK_ATTEMPTS` so the task dead-letters on
        the first occurrence. The corresponding draft flips to
        `state='error'` and the operator inspects the trace.

Adapters MUST raise these from inside `Provider.call` so the worker
sees them at the same boundary it already catches `Exception`.
"""

from __future__ import annotations


class OperatorActionRequired(RuntimeError):
    """Upstream rejected the request in a way only an admin can fix.

    Examples: 503 `model_not_found` from a router that doesn't know
    the requested model alias, 401 / 403 with a clearly revoked key,
    Bifrost 503 `no available credential for provider`. Raising this
    pauses the pipeline stage so a config swap (and only a config
    swap) is enough to bring it back — no retry storm in the
    meantime.
    """


class UpstreamUnavailable(RuntimeError):
    """Provider is reachable but cannot serve us right now.

    Examples: 502/504 from a gateway after the in-process retry
    budget is exhausted, timeouts from an overloaded inference host,
    rate-limit rejections that should clear on their own. The task
    waits with an exponential backoff and tries again.
    """
