"""Typed config with fixed app home directory."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .paths import Paths, home


# ── Config data ──────────────────────────────────────────────────


class ProviderConfig(BaseModel):
    type: str = "openai"
    endpoint: str = ""
    api_key: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)


class TranslationConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 16384
    reasoning_effort: str | None = None


class ContextAgentConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int | None = 8192
    reasoning_effort: str | None = None


class VisionAgentConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int | None = 4096
    reasoning_effort: str | None = None


class Config(BaseSettings):
    model_config = {"extra": "ignore"}

    models_dir: str = "models"
    default_target_lang: str = "vi"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    translation: TranslationConfig = TranslationConfig()
    context_agent: ContextAgentConfig = ContextAgentConfig()
    vision_agent: VisionAgentConfig = VisionAgentConfig()
    bubble_scope_imgsz: int = 640
    database_url: str = ""  # empty = SQLite default path, or "postgresql://..."


# ── Loading ──────────────────────────────────────────────────────


def _find_config_file(root: Path | None = None) -> Path:
    """Find config file: explicit root first, then CWD, then app home."""
    if root is not None:
        root_cfg = Path(root) / "config.toml"
        if root_cfg.exists():
            return root_cfg
    cwd_cfg = Path.cwd() / "config.toml"
    if cwd_cfg.exists():
        return cwd_cfg
    return home() / "config.toml"


def load_config(root: Path | None = None) -> tuple[Config, Paths]:
    """Load config from nearest config.toml (CWD → root → app home)."""
    config_path = _find_config_file(root)
    if config_path.exists():
        data = tomllib.loads(config_path.read_text())
        config = Config(**data)
    else:
        config = Config()

    if config_path.parent == Path.cwd():
        paths = Paths(Path.cwd())
    else:
        paths = Paths(root)
    models = Path(config.models_dir)
    if not models.is_absolute():
        config.models_dir = str(paths.root / models)
    for pcfg in config.providers.values():
        pcfg.api_key = os.path.expandvars(pcfg.api_key or "")
        pcfg.extra_headers = {
            k: os.path.expandvars(v) for k, v in pcfg.extra_headers.items()
        }
    return config, paths
