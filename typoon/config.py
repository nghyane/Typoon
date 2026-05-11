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
    # Maximum concurrent in-flight requests against this provider,
    # shared across every agent that resolves to it (context, translate,
    # vision). Sized by Little's law: at ~3s/call latency, 24 in-flight
    # ≈ 8 RPS, which sits inside paid Tier 1 budgets (OpenAI 500 RPM /
    # Anthropic Tier 1). Higher tiers tolerate more — bump to 64 for
    # 2000+ RPM accounts. Free tiers should drop to 1–2. Set to 0 to
    # disable the gate entirely (rely solely on retry + backoff).
    concurrency: int = 24


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

    `public_base_url` is the single public origin every client hits in
    production — both the SPA load and every API/file/CDN/upload path
    are fronted by it. In a Discord Activity deploy that's the DA host
    (`https://<app_id>.discordsays.com`) and Discord's URL Mappings
    forward `/`, `/api`, `/r2`, `/cdn`, `/t` to the matching upstreams.
    Browser clients running outside DA hit the same origin cross-
    origin; CORS allows it because the regex below covers DA hosts.
    Dev simply points at the local API and the SPA dev server proxies.
    """
    public_base_url: str = "http://localhost:8000"
    # Extra web origins to allow in CORS, beyond `public_base_url` and
    # the built-in `*.discordsays.com` regex. Needed when the SPA is
    # served from a different host than the public origin clients
    # primarily hit — e.g. CF Pages at `mangalocal.com` cross-origins
    # to the DA host `discordsays.com` to reach the API. Browsers
    # outside DA preflight from the SPA's own origin, which must be
    # whitelisted here or every fetch fails with "Disallowed CORS".
    extra_web_origins: list[str] = Field(default_factory=list)
    host:           str = "0.0.0.0"
    port:           int = 8000
    # Hosts allowed in the Host header. Behind Cloudflare Tunnel the
    # uvicorn server sees the tunnel's ingress hostname (e.g.
    # `api.mangalocal.com`), NOT `public_base_url` — Discord proxy +
    # tunnel both preserve the upstream's real hostname. Production
    # must set TRUSTED_HOSTS explicitly to that ingress; dev derives
    # from `public_base_url` + localhost. `["*"]` disables the check.
    trusted_hosts: list[str] = Field(default_factory=list)


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


class PublicStoreConfig(BaseModel):
    """Where rendered archives live (browser-facing)."""
    type:       str = "local"          # local | huggingface
    # huggingface
    hf_repo:    str = "nghyane/mcz-cdn"
    hf_token:   str = ""               # env: HF_TOKEN
    cdn_prefix: str = "https://927251094806098001.discordsays.com/cdn/t"


class PipelineStoreConfig(BaseModel):
    """Where pipeline blobs (prepared, masks) live for cross-worker sharing.

    Single-host: type=local, all workers share the same disk via the
    same LocalBlobStore. Multi-host: type=http pointing at the storage
    role's /api/blobs endpoint, typically reached via tailnet so the
    transport stays on the encrypted mesh.
    """
    type:           str = "local"      # local | http
    # http
    http_base_url:  str = ""           # e.g. http://100.72.203.52:8000
    http_api_token: str = ""           # env: PIPELINE_HTTP_TOKEN


class InboxConfig(BaseModel):
    """Short-lived chapter-zip inbox.

    Browser clients (web SPA, extension) PUT a packed chapter zip via
    pre-signed multipart URLs straight to this backend, sparing the
    engine's home upstream bandwidth. The worker then fetches the zip
    and prepares the chapter.

    Backends:
      - `local`: filesystem under `paths.artifacts/_inbox/`. Multipart
        is simulated; PUTs go to a sibling local API route. Dev only.
      - `s3`:    any S3-compatible endpoint (R2 / AWS S3 / B2 /
        MinIO / Wasabi). Set `s3_endpoint` to the provider URL.

    The bucket SHOULD have a 24h lifecycle rule on the prefix as a
    safety net for upload-aborted prefixes; the engine deletes
    successful uploads after ingest.

    The bucket MUST have CORS configured to expose the `ETag` header
    on PUT responses, otherwise the browser cannot complete the
    multipart flow:

        AllowedOrigins: <web origin> + chrome-extension://<id>
        AllowedMethods: PUT
        ExposeHeaders:  ETag
        MaxAgeSeconds:  3600
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
    """Per-user chapter quota for actions that consume LLM cost.

    A "chapter slot" is consumed by upload-with-start, manual /start,
    and /redo — never by idle upload or reads. Admins bypass entirely.

    Counters are time-windowed (last hour, last day) over rows in
    `chapter_consumes`; concurrent count is over chapters with a
    live task in flight, owned by the user.
    """
    chapters_per_hour:   int = 10
    chapters_per_day:    int = 50
    concurrent_chapters: int = 3


class Config(BaseSettings):
    model_config = {"extra": "ignore"}

    models_dir: str = "models"
    default_target_lang: str = "vi"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    translation: TranslationConfig = TranslationConfig()
    context_agent: ContextAgentConfig = ContextAgentConfig()
    vision_agent: VisionAgentConfig = VisionAgentConfig()
    bubble_scope_imgsz: int = 640
    # OCR backend choice. `"auto"` picks the first available backend in
    # the order google-lens → apple-vision → windows-ocr → tesseract.
    # Set explicitly (e.g. `"apple_vision"`) to pin a backend regardless
    # of what's installed. Japanese projects always use manga-ocr if it
    # is installed, ignoring this setting.
    ocr_backend: str = "auto"
    # Override the Google Lens endpoint. Empty → library default
    # (https://lensfrontend-pa.googleapis.com/v1/crupload). Setting this
    # to a Discord Activity proxy URL
    # (e.g. https://<app_id>.discordsays.com/lens/v1/crupload) routes
    # through Cloudflare's edge — measured 30–50% faster from APAC and
    # noticeably lower variance under load.
    lens_endpoint: str = ""
    # Postgres DSN. Required — engine refuses to start without it.
    database_url: str = ""
    server: ServerConfig = ServerConfig()
    auth:   AuthConfig   = AuthConfig()
    storage: StorageConfig = StorageConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()


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

    # Trusted hosts for the Host header. Comma-separated env or
    # auto-derive from public_base_url. Behind a tunnel/proxy the
    # ingress hostname differs from public_base_url and TRUSTED_HOSTS
    # MUST be set explicitly. "*" disables the check (dev only).
    env_hosts = os.environ.get("TRUSTED_HOSTS", "").strip()
    if env_hosts:
        config.server.trusted_hosts = [h.strip() for h in env_hosts.split(",") if h.strip()]
    elif not config.server.trusted_hosts:
        from urllib.parse import urlparse
        derived: list[str] = []
        host = urlparse(config.server.public_base_url).hostname
        if host:
            derived.append(host)
        # Always allow localhost so health probes / SSH tunnels still
        # work without explicit config.
        derived.extend(["localhost", "127.0.0.1"])
        # De-dup preserving order.
        seen: set[str] = set()
        config.server.trusted_hosts = [h for h in derived if not (h in seen or seen.add(h))]

    # Database — env wins over toml. Must be a postgresql:// DSN.
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
