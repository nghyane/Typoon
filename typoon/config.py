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


class ServerConfig(BaseModel):
    """API + web URLs.

    `public_api_url` is the public origin clients hit. `public_web_url`
    is where the SPA is served. Both are also used by CORS and by the
    SPA's OAuth bootstrap (the SPA owns the redirect_uri — engine never
    sees the web origin in the OAuth call itself).
    """
    public_api_url: str = "http://localhost:8000"
    public_web_url: str = "http://localhost:5173"
    host:           str = "0.0.0.0"
    port:           int = 8000


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
    # When gating is on, the engine fetches the public widget
    # (/guilds/{id}/widget.json) for the guild name + invite URL on
    # demand. The operator just needs to enable "Server Widget" in
    # Discord Server Settings — no extra config here.
    discord_guild_id:      str = ""
    # Discord role ID (snowflake) that grants admin privileges in the app.
    # Right-click role in Discord (with developer mode on) → Copy Role ID.
    # Empty = no app admin (engine refuses admin operations).
    admin_role_id:         str = ""
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
    # Postgres DSN. Required — RFC-005 dropped the SQLite fallback.
    database_url: str = ""
    server: ServerConfig = ServerConfig()
    auth:   AuthConfig   = AuthConfig()


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


def _load_dotenv(root: Path | None) -> None:
    """Load `.env` from `root` or CWD if present. OS env wins."""
    try:
        from dotenv import load_dotenv as _ld
    except ImportError:
        return
    for candidate in (
        Path(root) / ".env" if root else None,
        Path.cwd() / ".env",
    ):
        if candidate and candidate.exists():
            _ld(candidate, override=False)
            return


def load_config(root: Path | None = None) -> tuple[Config, Paths]:
    """Load config from nearest config.toml (CWD → root → app home).

    Also loads `.env` from CWD or `root` if present, so users don't have
    to `source .env` before every shell invocation. Existing OS env wins
    over `.env` (12-factor).
    """
    _load_dotenv(root)
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

    # Server URLs — env wins over toml.
    config.server.public_api_url = os.environ.get(
        "PUBLIC_API_URL", config.server.public_api_url,
    ).rstrip("/")
    config.server.public_web_url = os.environ.get(
        "PUBLIC_WEB_URL", config.server.public_web_url,
    ).rstrip("/")
    if env_port := os.environ.get("TYPOON_PORT"):
        try:
            config.server.port = int(env_port)
        except ValueError:
            pass

    # Database — env wins over toml. RFC-005 requires a postgresql:// DSN.
    config.database_url = os.environ.get("DATABASE_URL", config.database_url)
    if not config.database_url:
        raise RuntimeError(
            "DATABASE_URL is required (RFC-005). "
            "Set it in .env or config.toml, e.g. "
            "postgresql://typoon:typoon@localhost:5432/typoon"
        )
    if not config.database_url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError(
            f"DATABASE_URL must be postgresql://… — got {config.database_url!r}. "
            f"SQLite is not supported (RFC-005)."
        )

    # Auth: env vars take precedence over config.toml. Secrets should never
    # land in toml. JWT secret is auto-generated and persisted to disk if
    # missing so dev sessions don't get invalidated on restart.
    config.auth.discord_client_id     = os.environ.get("DISCORD_CLIENT_ID",     config.auth.discord_client_id)
    config.auth.discord_client_secret = os.environ.get("DISCORD_CLIENT_SECRET", config.auth.discord_client_secret)
    config.auth.discord_guild_id      = os.environ.get("DISCORD_GUILD_ID",      config.auth.discord_guild_id)
    config.auth.admin_role_id         = os.environ.get("DISCORD_ADMIN_ROLE_ID", config.auth.admin_role_id)
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
