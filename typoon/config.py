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


class ProviderConfig(BaseModel):
    type: str = "openai"
    endpoint: str = ""
    api_key: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    # "chat" = /v1/chat/completions; "responses" = /v1/responses.
    api_kind: str = "chat"
    # In-flight cap shared across agents resolving to this provider.
    # 0 disables the gate.
    concurrency: int = 24


class TranslationConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 16384
    reasoning_effort: str | None = None


class VisionAgentConfig(BaseModel):
    """Vision provider — drives both the per-chapter storyboard context pass
    (`stages.scan_context`) and on-demand `look_at` calls during translation."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int | None = 4096
    reasoning_effort: str | None = None


class ServerConfig(BaseModel):
    """Single public origin fronting SPA + API + CDN + upload paths."""
    public_base_url: str = "http://localhost:8000"
    # Additional CORS origins beyond public_base_url and the built-in
    # *.discordsays.com regex (e.g. when the SPA is served from a
    # different host than the API).
    extra_web_origins: list[str] = Field(default_factory=list)
    host:           str = "0.0.0.0"
    port:           int = 8000
    # Host-header allowlist. Must include the tunnel/proxy ingress
    # hostname in production (it can differ from public_base_url).
    # ["*"] disables the check.
    trusted_hosts: list[str] = Field(default_factory=list)


class AuthConfig(BaseModel):
    """Discord OAuth + JWT. Secrets read from env in load_config()."""
    discord_client_id:     str = ""
    discord_client_secret: str = ""
    # Empty disables membership gating.
    discord_guild_id:      str = ""
    # Empty = engine refuses admin operations.
    admin_role_id:         str = ""
    # MUST be set in production; auto-generated for dev.
    jwt_secret:            str = ""
    session_days:          int = 30


class PublicStoreConfig(BaseModel):
    """Where rendered archives live (browser-facing)."""
    type:       str = "local"          # local | huggingface
    # huggingface
    hf_repo:    str = "nghyane/mcz-cdn"
    hf_token:   str = ""               # env: HF_TOKEN
    cdn_prefix: str = "https://927251094806098001.discordsays.com/cdn/t"


class PipelineStoreConfig(BaseModel):
    """Cross-worker blob storage. local=shared disk, http=storage role."""
    type:           str = "local"      # local | http
    http_base_url:  str = ""           # e.g. http://100.72.203.52:8000
    http_api_token: str = ""           # env: PIPELINE_HTTP_TOKEN


class InboxConfig(BaseModel):
    """Short-lived chapter-zip inbox.

    Browser clients PUT zips via presigned multipart URLs straight to
    the backend; the worker fetches them after.

    For s3 backends the bucket MUST expose the ETag header on PUT
    (CORS ExposeHeaders: ETag), otherwise the multipart flow breaks.
    A 24h lifecycle rule on the prefix is recommended.
    """
    type:                 str = "local"      # local | s3
    # s3-compatible
    s3_endpoint:          str = ""
    s3_bucket:            str = ""
    s3_region:            str = "auto"
    s3_access_key_id:     str = ""
    s3_secret_access_key: str = ""
    s3_prefix:            str = "tmp/"


class StorageConfig(BaseModel):
    """Two-tier storage: public (browser) + pipeline (workers).

    `archive_path_salt` is the HMAC salt for unguessable render archive
    keys. Empty → derived from auth.jwt_secret in load_config().
    """
    public:            PublicStoreConfig   = Field(default_factory=PublicStoreConfig)
    pipeline:          PipelineStoreConfig = Field(default_factory=PipelineStoreConfig)
    inbox:             InboxConfig         = Field(default_factory=InboxConfig)
    archive_path_salt: str = ""


class RateLimitConfig(BaseModel):
    """Per-user chapter slot quota. Consumed by draft/render-create."""
    chapters_per_hour: int = 10
    chapters_per_day:  int = 50


class DatabaseConfig(BaseModel):
    """Postgres pool sizing.

    statement_cache_size=0 sidesteps asyncpg's prepared-statement
    cache, which has surfaced _get_statement races under concurrent
    first requests.
    """
    pool_min_size:        int = 2
    pool_max_size:        int = 10
    statement_cache_size: int = 0


class Config(BaseSettings):
    model_config = {"extra": "ignore"}

    models_dir: str = "models"
    default_target_lang: str = "vi"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    translation: TranslationConfig = TranslationConfig()
    vision_agent: VisionAgentConfig = VisionAgentConfig()
    bubble_scope_imgsz: int = 640
    # "auto" tries google-lens → apple-vision → windows-ocr → tesseract.
    # Japanese always uses manga-ocr if installed.
    ocr_backend: str = "auto"
    # Empty → google-lens default endpoint. Override e.g. to route
    # through a Discord Activity proxy.
    lens_endpoint: str = ""
    # Postgres DSN. Required — engine refuses to start without it.
    database_url: str = ""
    server: ServerConfig = ServerConfig()
    auth:   AuthConfig   = AuthConfig()
    storage: StorageConfig = StorageConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    database: DatabaseConfig = DatabaseConfig()


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

    config.server.public_base_url = os.environ.get(
        "PUBLIC_BASE_URL", config.server.public_base_url,
    ).rstrip("/")
    env_extra = os.environ.get("EXTRA_WEB_ORIGINS", "").strip()
    if env_extra:
        config.server.extra_web_origins = [
            o.strip().rstrip("/") for o in env_extra.split(",") if o.strip()
        ]
    if env_port := os.environ.get("TYPOON_PORT"):
        try:
            config.server.port = int(env_port)
        except ValueError:
            pass

    env_hosts = os.environ.get("TRUSTED_HOSTS", "").strip()
    if env_hosts:
        config.server.trusted_hosts = [h.strip() for h in env_hosts.split(",") if h.strip()]
    elif not config.server.trusted_hosts:
        from urllib.parse import urlparse
        derived: list[str] = []
        host = urlparse(config.server.public_base_url).hostname
        if host:
            derived.append(host)
        derived.extend(["localhost", "127.0.0.1"])
        seen: set[str] = set()
        config.server.trusted_hosts = [h for h in derived if not (h in seen or seen.add(h))]

    config.database_url = os.environ.get("DATABASE_URL", config.database_url)
    if not config.database_url:
        raise RuntimeError(
            "DATABASE_URL is required. "
            "Set it in .env or config.toml, e.g. "
            "postgresql://typoon:typoon@localhost:5432/typoon"
        )
    if not config.database_url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError(
            f"DATABASE_URL must be postgresql://… — got {config.database_url!r}."
        )

    def _env_int(name: str, current: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return current
        try:
            return int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be int, got {raw!r}") from exc

    config.database.pool_min_size = _env_int(
        "DB_POOL_MIN", config.database.pool_min_size,
    )
    config.database.pool_max_size = _env_int(
        "DB_POOL_MAX", config.database.pool_max_size,
    )
    config.database.statement_cache_size = _env_int(
        "DB_STATEMENT_CACHE", config.database.statement_cache_size,
    )
    if config.database.pool_min_size > config.database.pool_max_size:
        raise RuntimeError(
            "database.pool_min_size > pool_max_size: "
            f"{config.database.pool_min_size} > {config.database.pool_max_size}"
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
        # Auto-generate for dev. In production, JWT_SECRET MUST be set
        # explicitly — otherwise every restart invalidates every active
        # session. Marker env: TYPOON_ENV=production.
        if os.environ.get("TYPOON_ENV", "").lower() == "production":
            raise RuntimeError(
                "JWT_SECRET environment variable is required when "
                "TYPOON_ENV=production. Generate one with "
                "`python -c \"import secrets; print(secrets.token_urlsafe(64))\"` "
                "and persist it (e.g. systemd EnvironmentFile)."
            )
        config.auth.jwt_secret = _ensure_jwt_secret(paths.root)

    # Storage: env wins over toml. Salt defaults to a derivation of
    # `jwt_secret` so dev doesn't need a separate env var; prod can
    # rotate it independently via ARCHIVE_PATH_SALT.
    #
    # Naming: storage backend vars are namespaced by backend
    # (HF_*, R2_*, PIPELINE_*, PUBLIC_*) — no `TYPOON_` prefix on
    # storage config. The `TYPOON_` prefix is reserved for app-level
    # deployment markers (TYPOON_ENV, TYPOON_PORT, TYPOON_API_ROLE).
    config.storage.public.type           = os.environ.get("PUBLIC_STORE_TYPE",      config.storage.public.type)
    config.storage.public.hf_repo        = os.environ.get("HF_REPO",                config.storage.public.hf_repo)
    config.storage.public.hf_token       = os.environ.get("HF_TOKEN",               config.storage.public.hf_token)
    config.storage.public.cdn_prefix     = os.environ.get("HF_CDN_PREFIX",          config.storage.public.cdn_prefix)
    config.storage.pipeline.type         = os.environ.get("PIPELINE_STORE_TYPE",    config.storage.pipeline.type)
    config.storage.pipeline.http_base_url  = os.environ.get("PIPELINE_HTTP_BASE_URL", config.storage.pipeline.http_base_url)
    config.storage.pipeline.http_api_token = os.environ.get("PIPELINE_HTTP_TOKEN",    config.storage.pipeline.http_api_token)
    config.storage.archive_path_salt     = os.environ.get(
        "ARCHIVE_PATH_SALT", config.storage.archive_path_salt,
    )
    if not config.storage.archive_path_salt:
        import hashlib
        config.storage.archive_path_salt = hashlib.sha256(
            (config.auth.jwt_secret + ":archive").encode()
        ).hexdigest()

    # Inbox (browser-direct chapter zip upload). All env-only so the
    # secret access key never lands in toml. Type auto-flips to `s3`
    # the moment the four required S3 fields are present; otherwise
    # it falls back to the local-on-disk inbox simulator (dev only).
    config.storage.inbox.s3_endpoint          = os.environ.get("INBOX_S3_ENDPOINT",          config.storage.inbox.s3_endpoint)
    config.storage.inbox.s3_bucket            = os.environ.get("INBOX_S3_BUCKET",            config.storage.inbox.s3_bucket)
    config.storage.inbox.s3_region            = os.environ.get("INBOX_S3_REGION",            config.storage.inbox.s3_region)
    config.storage.inbox.s3_access_key_id     = os.environ.get("INBOX_S3_ACCESS_KEY_ID",     config.storage.inbox.s3_access_key_id)
    config.storage.inbox.s3_secret_access_key = os.environ.get("INBOX_S3_SECRET_ACCESS_KEY", config.storage.inbox.s3_secret_access_key)
    config.storage.inbox.s3_prefix            = os.environ.get("INBOX_S3_PREFIX",            config.storage.inbox.s3_prefix)
    if (config.storage.inbox.s3_endpoint
            and config.storage.inbox.s3_bucket
            and config.storage.inbox.s3_access_key_id
            and config.storage.inbox.s3_secret_access_key):
        config.storage.inbox.type = "s3"

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
