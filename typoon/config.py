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
    extra_headers: dict[str, str] = Field(default_factory=dict)
    # Cloudflare Gateway: set cf-aig-authorization header
    # e.g. extra_headers = { "cf-aig-authorization": "Bearer $TOKEN" }


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

    # Determine effective root: if CWD has config.toml, resolve relative to CWD
    if config_path.parent == Path.cwd():
        paths = Paths(Path.cwd())
    else:
        paths = Paths(root)
    # Resolve models_dir relative to home
    models = Path(config.models_dir)
    if not models.is_absolute():
        config.models_dir = str(paths.root / models)
    # Expand env vars in provider config (e.g. $CF_AIG_TOKEN in extra_headers)
    for pcfg in config.providers.values():
        pcfg.api_key = os.path.expandvars(pcfg.api_key or "")
        pcfg.extra_headers = {
            k: os.path.expandvars(v) for k, v in pcfg.extra_headers.items()
        }
    return config, paths
