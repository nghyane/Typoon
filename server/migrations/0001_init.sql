-- +goose Up
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE users (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  display_name TEXT,
  avatar_url TEXT,
  is_admin   BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE flows (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  kind          TEXT NOT NULL CHECK (kind IN ('auth_discord', 'payos_return', 'discord_activity')),
  status        TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'expired', 'failed')),
  user_id       UUID REFERENCES users(id),
  state_hash    TEXT UNIQUE,
  context_json  JSONB NOT NULL DEFAULT '{}'::jsonb,
  result_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
  expires_at    TIMESTAMPTZ NOT NULL,
  completed_at  TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_flows_pending_expiry ON flows(status, expires_at);
CREATE INDEX idx_flows_user_created ON flows(user_id, created_at DESC);

CREATE TABLE oauth_accounts (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider            TEXT NOT NULL CHECK (provider IN ('discord')),
  provider_account_id TEXT NOT NULL,
  username            TEXT,
  avatar_url          TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, provider_account_id),
  UNIQUE (user_id, provider)
);

CREATE TABLE auth_sessions (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash   TEXT NOT NULL UNIQUE,
  origin       TEXT NOT NULL,
  user_agent   TEXT,
  ip_hash      TEXT,
  expires_at   TIMESTAMPTZ NOT NULL,
  revoked_at   TIMESTAMPTZ,
  last_seen_at TIMESTAMPTZ,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_auth_sessions_user ON auth_sessions(user_id, created_at DESC);
CREATE INDEX idx_auth_sessions_active ON auth_sessions(token_hash, expires_at) WHERE revoked_at IS NULL;

CREATE TABLE llm_providers (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('openai', 'openrouter', 'anthropic', 'local', 'custom')),
  base_url    TEXT NOT NULL,
  api_key_ref TEXT NOT NULL,
  enabled     BOOLEAN NOT NULL DEFAULT true,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE llm_profiles (
  id            TEXT PRIMARY KEY,
  provider_id   TEXT NOT NULL REFERENCES llm_providers(id),
  name          TEXT NOT NULL,
  model         TEXT NOT NULL,
  protocol      TEXT NOT NULL CHECK (protocol IN ('openai_responses', 'openai_chat_completions', 'anthropic_messages')),
  endpoint_path TEXT NOT NULL,
  params_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
  timeout_ms    INTEGER NOT NULL DEFAULT 60000 CHECK (timeout_ms > 0),
  enabled       BOOLEAN NOT NULL DEFAULT true,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_llm_profiles_provider ON llm_profiles(provider_id);

CREATE TABLE llm_policies (
  purpose            TEXT PRIMARY KEY,
  profile_chain_json JSONB NOT NULL,
  enabled            BOOLEAN NOT NULL DEFAULT true,
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (jsonb_typeof(profile_chain_json) = 'array')
);

CREATE TABLE llm_attempts (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  purpose       TEXT NOT NULL,
  provider_id   TEXT REFERENCES llm_providers(id),
  profile_id    TEXT REFERENCES llm_profiles(id),
  status        TEXT NOT NULL CHECK (status IN ('success', 'failed')),
  latency_ms    INTEGER NOT NULL CHECK (latency_ms >= 0),
  error_code    TEXT,
  input_tokens  INTEGER CHECK (input_tokens >= 0),
  output_tokens INTEGER CHECK (output_tokens >= 0),
  total_tokens  INTEGER CHECK (total_tokens >= 0),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_llm_attempts_purpose_created ON llm_attempts(purpose, created_at DESC);
CREATE INDEX idx_llm_attempts_profile_created ON llm_attempts(profile_id, created_at DESC);

CREATE TABLE coin_packages (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name       TEXT NOT NULL,
  xu_amount  INTEGER NOT NULL CHECK (xu_amount > 0),
  bonus_xu   INTEGER NOT NULL DEFAULT 0 CHECK (bonus_xu >= 0),
  price_vnd  INTEGER NOT NULL CHECK (price_vnd > 0),
  enabled    BOOLEAN NOT NULL DEFAULT true,
  sort_order INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE payment_orders (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID NOT NULL REFERENCES users(id),
  provider            TEXT NOT NULL CHECK (provider IN ('payos')),
  provider_order_code TEXT NOT NULL,
  coin_package_id     UUID NOT NULL REFERENCES coin_packages(id),
  amount_vnd          INTEGER NOT NULL CHECK (amount_vnd > 0),
  xu_amount           INTEGER NOT NULL CHECK (xu_amount > 0),
  status              TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'paid', 'cancelled', 'expired', 'failed')),
  idempotency_key     TEXT,
  checkout_url        TEXT,
  paid_at             TIMESTAMPTZ,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, provider_order_code)
);

CREATE UNIQUE INDEX idx_payment_orders_user_idempotency
  ON payment_orders(user_id, idempotency_key)
  WHERE idempotency_key IS NOT NULL;

CREATE INDEX idx_payment_orders_user_created ON payment_orders(user_id, created_at DESC);

CREATE TABLE payment_webhook_events (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  provider            TEXT NOT NULL CHECK (provider IN ('payos')),
  provider_order_code TEXT NOT NULL,
  payload_hash        TEXT NOT NULL,
  payload_json        JSONB NOT NULL,
  signature_valid     BOOLEAN NOT NULL,
  processed_at        TIMESTAMPTZ,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, payload_hash)
);

CREATE INDEX idx_payment_webhooks_order ON payment_webhook_events(provider, provider_order_code, created_at DESC);

CREATE TABLE wallet_accounts (
  user_id      UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  available_xu INTEGER NOT NULL DEFAULT 0 CHECK (available_xu >= 0),
  held_xu      INTEGER NOT NULL DEFAULT 0 CHECK (held_xu >= 0),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE wallet_ledger (
  id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                  UUID NOT NULL REFERENCES users(id),
  kind                     TEXT NOT NULL CHECK (kind IN ('topup', 'hold', 'capture', 'release', 'refund', 'admin_adjustment')),
  available_delta_xu       INTEGER NOT NULL,
  held_delta_xu            INTEGER NOT NULL,
  balance_available_after  INTEGER NOT NULL CHECK (balance_available_after >= 0),
  balance_held_after       INTEGER NOT NULL CHECK (balance_held_after >= 0),
  reference_type           TEXT NOT NULL CHECK (reference_type IN ('payment_order', 'translation_session', 'admin_action')),
  reference_id             TEXT NOT NULL,
  note                     TEXT,
  created_by_admin_id      UUID REFERENCES users(id),
  created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (user_id, kind, reference_type, reference_id)
);

CREATE INDEX idx_wallet_ledger_user_created ON wallet_ledger(user_id, created_at DESC);
CREATE INDEX idx_wallet_ledger_reference ON wallet_ledger(reference_type, reference_id);

CREATE TABLE translation_sessions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES users(id),
  content_key   TEXT NOT NULL,
  work_id       TEXT,
  segment_id    TEXT,
  source_lang   TEXT,
  target_lang   TEXT NOT NULL,
  mode          TEXT NOT NULL CHECK (mode IN ('draft_free', 'refined')),
  state         TEXT NOT NULL DEFAULT 'created' CHECK (state IN ('created', 'drafting', 'refining', 'done', 'error', 'expired', 'cancelled')),
  xu_state      TEXT NOT NULL DEFAULT 'free' CHECK (xu_state IN ('free', 'entitled', 'held', 'captured', 'released', 'waived')),
  price_xu      INTEGER NOT NULL DEFAULT 0 CHECK (price_xu >= 0),
  page_count    INTEGER CHECK (page_count >= 0),
  unit_count    INTEGER CHECK (unit_count >= 0),
  error_code    TEXT,
  error_message TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at   TIMESTAMPTZ
);

CREATE INDEX idx_translation_sessions_user_created ON translation_sessions(user_id, created_at DESC);
CREATE INDEX idx_translation_sessions_content ON translation_sessions(content_key, target_lang);

CREATE TABLE translation_holds (
  session_id  UUID PRIMARY KEY REFERENCES translation_sessions(id) ON DELETE CASCADE,
  user_id     UUID NOT NULL REFERENCES users(id),
  amount_xu   INTEGER NOT NULL CHECK (amount_xu > 0),
  state       TEXT NOT NULL DEFAULT 'held' CHECK (state IN ('held', 'captured', 'released')),
  held_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at TIMESTAMPTZ
);

CREATE INDEX idx_translation_holds_user_state ON translation_holds(user_id, state);

CREATE TABLE translation_entitlements (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES users(id),
  content_key TEXT NOT NULL,
  target_lang TEXT NOT NULL,
  session_id  UUID NOT NULL REFERENCES translation_sessions(id),
  granted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (user_id, content_key, target_lang)
);

CREATE TABLE translation_refine_windows (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id     UUID NOT NULL REFERENCES translation_sessions(id) ON DELETE CASCADE,
  window_index   INTEGER NOT NULL CHECK (window_index >= 0),
  request_json   JSONB NOT NULL,
  response_json  JSONB,
  status         TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'error')),
  llm_profile_id TEXT REFERENCES llm_profiles(id),
  latency_ms     INTEGER CHECK (latency_ms >= 0),
  error_code     TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at   TIMESTAMPTZ,
  UNIQUE (session_id, window_index)
);

CREATE INDEX idx_translation_refine_windows_session ON translation_refine_windows(session_id, window_index);

CREATE TABLE translation_refine_cache (
  cache_key      TEXT PRIMARY KEY,
  target_lang    TEXT NOT NULL,
  source_hash    TEXT NOT NULL,
  draft_hash     TEXT NOT NULL,
  target_text    TEXT NOT NULL,
  llm_profile_id TEXT REFERENCES llm_profiles(id),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE translation_price_rules (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  mode        TEXT NOT NULL CHECK (mode IN ('refined')),
  price_xu    INTEGER NOT NULL CHECK (price_xu >= 0),
  enabled     BOOLEAN NOT NULL DEFAULT true,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- +goose Down
DROP TABLE IF EXISTS translation_price_rules CASCADE;
DROP TABLE IF EXISTS translation_refine_cache CASCADE;
DROP TABLE IF EXISTS translation_refine_windows CASCADE;
DROP TABLE IF EXISTS translation_entitlements CASCADE;
DROP TABLE IF EXISTS translation_holds CASCADE;
DROP TABLE IF EXISTS translation_sessions CASCADE;
DROP TABLE IF EXISTS wallet_ledger CASCADE;
DROP TABLE IF EXISTS wallet_accounts CASCADE;
DROP TABLE IF EXISTS payment_webhook_events CASCADE;
DROP TABLE IF EXISTS payment_orders CASCADE;
DROP TABLE IF EXISTS coin_packages CASCADE;
DROP TABLE IF EXISTS llm_attempts CASCADE;
DROP TABLE IF EXISTS llm_policies CASCADE;
DROP TABLE IF EXISTS llm_profiles CASCADE;
DROP TABLE IF EXISTS llm_providers CASCADE;
DROP TABLE IF EXISTS auth_sessions CASCADE;
DROP TABLE IF EXISTS oauth_accounts CASCADE;
DROP TABLE IF EXISTS flows CASCADE;
DROP TABLE IF EXISTS users CASCADE;
