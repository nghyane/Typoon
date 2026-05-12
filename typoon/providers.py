"""Provider factory — builds LLM providers from config.

Concurrency model:

A `[providers.<name>]` block in config.toml represents a real upstream
endpoint (OpenRouter free tier, OpenAI account, etc.). Rate limits live
on that endpoint, not on the agent that uses it. So the semaphore lives
on the **provider name** — every agent (context, translate, vision)
that resolves to the same provider name shares the same in-flight cap.

The shared semaphore matters most for the translate stage, which fans
out windows in parallel via asyncio.gather(): without it, a single
chapter with 6 windows × 3 concurrent chapters = 18 simultaneous calls
would shred a small-quota provider regardless of any per-worker cap.
"""

from __future__ import annotations

import asyncio
import os
import threading

from .config import Config, ProviderConfig
from .llm.ir import CallResponse, Message, Provider, ToolDef


# Process-wide registry: provider-name → semaphore. Lazy init on first
# acquire so we bind to whichever event loop the worker is running.
# The lock guards registry mutation only; semaphore acquire itself is
# loop-bounded and async.
_SEM_LOCK = threading.Lock()
_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _shared_semaphore(name: str, limit: int) -> asyncio.Semaphore:
    with _SEM_LOCK:
        sem = _SEMAPHORES.get(name)
        if sem is None:
            sem = asyncio.Semaphore(max(1, int(limit)))
            _SEMAPHORES[name] = sem
        return sem


class _RateLimitedProvider:
    """Decorator that gates an inner provider with a shared semaphore.

    Multiple `_RateLimitedProvider` instances built for the same
    `provider_name` share one `asyncio.Semaphore` from the registry —
    that's how context/translate/vision agents pointing at the same
    upstream endpoint cooperate on the same in-flight cap.

    `limit <= 0` disables the gate entirely; the call passes straight
    through to the inner provider. Useful for enterprise tiers where
    the provider's own rate limiter and the SDK's retry/backoff are
    enough.
    """

    __slots__ = ("_inner", "_provider_name", "_limit", "_sem")

    def __init__(self, inner: Provider, provider_name: str, limit: int) -> None:
        self._inner = inner
        self._provider_name = provider_name
        self._limit = int(limit)
        self._sem: asyncio.Semaphore | None = None

    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        if self._limit <= 0:
            return await self._inner.call(messages, tools)
        if self._sem is None:
            self._sem = _shared_semaphore(self._provider_name, self._limit)
        async with self._sem:
            return await self._inner.call(messages, tools)


def make_translation_provider(config: Config) -> Provider:
    name = config.translation.provider
    pcfg = config.providers.get(name)
    if pcfg is None:
        raise ValueError(f"Provider '{name}' not found in [providers]")
    return _build_provider(
        name, pcfg, config.translation.model, config.translation.reasoning_effort,
        max_tokens=config.translation.max_tokens,
    )


def make_vision_provider(config: Config) -> Provider:
    name = config.vision_agent.provider
    pcfg = config.providers.get(name)
    if pcfg is None:
        raise ValueError(f"Provider '{name}' not found in [providers]")
    return _build_provider(
        name, pcfg, config.vision_agent.model, config.vision_agent.reasoning_effort,
        max_tokens=config.vision_agent.max_tokens,
    )


def _resolve_api_key(pcfg: ProviderConfig) -> str | None:
    """Return API key: explicit value (even empty string), env var, or fallback."""
    # Explicitly set (including empty string for no-auth gateways)
    if pcfg.api_key is not None:
        return pcfg.api_key
    env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
    env_key = env_map.get(pcfg.type)
    if env_key and os.environ.get(env_key):
        return os.environ[env_key]
    return "not-needed"


def _build_provider(
    provider_name: str,
    pcfg: ProviderConfig,
    model: str,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> Provider:
    # Keep empty string "" as-is for CF Gateway — SDK skips Authorization when empty.
    api_key = _resolve_api_key(pcfg)
    extra = pcfg.extra_headers if pcfg.extra_headers else None
    match pcfg.type:
        case "anthropic":
            from .llm.anthropic import AnthropicProvider
            kwargs: dict = {"base_url": pcfg.endpoint.removesuffix("/v1"), "api_key": api_key, "model": model}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            inner: Provider = AnthropicProvider(**kwargs)
        case "gemini":
            from .llm.gemini import GeminiProvider
            inner = GeminiProvider(api_key=api_key, model=model)
        case _:
            from .llm.openai import OpenAIProvider
            kwargs = {
                "base_url": pcfg.endpoint or None,
                "api_key": api_key,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "extra_headers": extra,
                "api_kind": pcfg.api_kind,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            inner = OpenAIProvider(**kwargs)

    return _RateLimitedProvider(inner, provider_name, pcfg.concurrency)
