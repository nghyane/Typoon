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


class AuthConfig(BaseModel):
    """Discord OAuth + JWT session config.

    Secrets (client_secret, jwt_secret) come from environment variables
    in load_config() — never the toml. The SPA owns the OAuth redirect
    URI (it points at the web origin's /auth/callback), so the engine
    does not store one.
    """
    discord_client_id:     str = ""
    discord_client_secret: str = ""
    # Optional gating: if set, user must be a member of this guild snowflake.
    discord_guild_id:      str = ""
    # Friendly invite URL the engine surfaces when the gate fails. The
    # operator pastes a stable invite from Discord (Server → Invite People
    # → Edit invite link → Never expire). Used in the 403 message and the
    # /api/auth/config endpoint so the SPA can show a "Join the server"
    # button on /login.
    discord_invite_url:    str = ""
    # Optional bootstrap admin: this discord_id is promoted to tier='admin'
    # on first login. Empty string = no auto-promotion.
    bootstrap_discord_id:  str = ""
    # JWT signing key. MUST be set in production. Auto-generated for dev.
    jwt_secret:            str = ""
    # Session lifetime (days)
    session_days:          int = 30


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
    auth:        AuthConfig = AuthConfig()


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

    # Auth: env vars take precedence over config.toml. Secrets should never
    # land in toml. JWT secret is auto-generated and persisted to disk if
    # missing so dev sessions don't get invalidated on restart.
    config.auth.discord_client_id     = os.environ.get("DISCORD_CLIENT_ID",     config.auth.discord_client_id)
    config.auth.discord_client_secret = os.environ.get("DISCORD_CLIENT_SECRET", config.auth.discord_client_secret)
    config.auth.discord_guild_id      = os.environ.get("DISCORD_GUILD_ID",      config.auth.discord_guild_id)
    config.auth.discord_invite_url    = os.environ.get("DISCORD_INVITE_URL",    config.auth.discord_invite_url)
    config.auth.bootstrap_discord_id  = os.environ.get("TYPOON_BOOTSTRAP_DISCORD_ID", config.auth.bootstrap_discord_id)
    config.auth.jwt_secret            = os.environ.get("JWT_SECRET",            config.auth.jwt_secret)
    if not config.auth.jwt_secret:
        config.auth.jwt_secret = _ensure_jwt_secret(paths.root)

    return config, paths


def _ensure_jwt_secret(root: Path) -> str:
    """Persist a random JWT secret to disk so dev restarts don't log users
    out. Production should always set JWT_SECRET via env."""
    import secrets
    secret_path = root / ".jwt_secret"
    if secret_path.exists():
        return secret_path.read_text().strip()
    root.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(48)
    secret_path.write_text(token)
    secret_path.chmod(0o600)
    return token
