"""Provider factory — builds LLM providers from config."""

from __future__ import annotations

import os

from .config import Config, ProviderConfig
from .llm.ir import Provider


def make_translation_provider(config: Config) -> Provider:
    name = config.translation.provider
    pcfg = config.providers.get(name)
    if pcfg is None:
        raise ValueError(f"Provider '{name}' not found in [providers]")
    return _build_provider(pcfg, config.translation.model, config.translation.reasoning_effort)


def make_context_provider(config: Config) -> Provider:
    name = config.context_agent.provider
    pcfg = config.providers.get(name)
    if pcfg is None:
        raise ValueError(f"Provider '{name}' not found in [providers]")
    return _build_provider(pcfg, config.context_agent.model, config.context_agent.reasoning_effort)


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


def _build_provider(pcfg: ProviderConfig, model: str, reasoning_effort: str | None = None) -> Provider:
    # Keep empty string "" as-is for CF Gateway — SDK skips Authorization when empty.
    api_key = _resolve_api_key(pcfg)
    extra = pcfg.extra_headers if pcfg.extra_headers else None
    match pcfg.type:
        case "anthropic":
            from .llm.anthropic import AnthropicProvider
            return AnthropicProvider(base_url=pcfg.endpoint.removesuffix("/v1"), api_key=api_key, model=model)
        case "gemini":
            from .llm.gemini import GeminiProvider
            return GeminiProvider(api_key=api_key, model=model)
        case _:
            from .llm.openai import OpenAIProvider
            return OpenAIProvider(
                base_url=pcfg.endpoint or None,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
                extra_headers=extra,
            )
