"""Typed config with fixed app home directory.

All paths resolve from TYPOON_HOME (~/.typoon by default).
Override with env TYPOON_HOME.

    ~/.typoon/
    ├── config.toml
    ├── typoon.db
    ├── models/
    ├── cache/
    ├── output/
    └── projects/
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── App home ─────────────────────────────────────────────────────

def home() -> Path:
    """Fixed app data directory. Never depends on CWD."""
    return Path(os.environ.get("TYPOON_HOME", "~/.typoon")).expanduser()


class Paths:
    """All app paths resolved from home."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or home()).resolve()

    @property
    def config_file(self) -> Path: return self.root / "config.toml"
    @property
    def db(self) -> Path: return self.root / "typoon.db"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def cache(self) -> Path: return self.root / "cache"
    @property
    def output(self) -> Path: return self.root / "output"
    @property
    def projects(self) -> Path: return self.root / "projects"

    def ensure(self) -> None:
        """Create all directories."""
        for d in (self.root, self.cache, self.output, self.projects):
            d.mkdir(parents=True, exist_ok=True)


# ── Config data ──────────────────────────────────────────────────


class ProviderConfig(BaseModel):
    type: str = "openai"
    endpoint: str = ""
    api_key: str | None = None


class TranslationConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    reasoning_effort: str | None = None


class ContextAgentConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    reasoning_effort: str | None = None


class Config(BaseSettings):
    model_config = {"extra": "ignore"}

    models_dir: str = "models"
    default_target_lang: str = "vi"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    translation: TranslationConfig = TranslationConfig()
    context_agent: ContextAgentConfig = ContextAgentConfig()


# ── Loading ──────────────────────────────────────────────────────


def load_config(root: Path | None = None) -> tuple[Config, Paths]:
    """Load config from app home. Returns (config, paths)."""
    paths = Paths(root)
    if paths.config_file.exists():
        data = tomllib.loads(paths.config_file.read_text())
        config = Config(**data)
    else:
        config = Config()
    # Resolve models_dir relative to home
    models = Path(config.models_dir)
    if not models.is_absolute():
        config.models_dir = str(paths.root / models)
    return config, paths
