SHELL := /bin/bash

ENV_FILE ?= .env
ARGS ?=

define with_env
set -a; [ ! -f "$(ENV_FILE)" ] || source "$(ENV_FILE)"; set +a;
endef

.PHONY: run dev build test db-reset db-seed

run:
	$(with_env) exec go run ./cmd/api $(ARGS)

dev:
	$(with_env) exec go run ./cmd/api $(ARGS)

build:
	go build -o bin/api ./cmd/api

test:
	go test ./...

db-reset:
	dropdb --if-exists typoon_test && createdb typoon_test
	sed '/^-- +goose Down/,$$d' migrations/0001_init.sql | psql "postgres:///typoon_test?sslmode=disable"
	psql "postgres:///typoon_test?sslmode=disable" -c "\
		INSERT INTO coin_packages (name, xu_amount, bonus_xu, price_vnd, sort_order) VALUES \
		('Gói 100 xu', 100, 0, 10000, 1), \
		('Gói 500 xu', 500, 50, 45000, 2), \
		('Gói 1000 xu', 1000, 150, 85000, 3), \
		('Gói 5000 xu', 5000, 1000, 400000, 4); \
		INSERT INTO users (id, display_name, is_admin) VALUES (gen_random_uuid(), 'Test User', false); \
		INSERT INTO wallet_accounts (user_id, available_xu, held_xu) SELECT id, 0, 0 FROM users LIMIT 1; \
		INSERT INTO llm_providers (id, name, kind, base_url, api_key_ref) VALUES ('openai', 'OpenAI', 'openai', 'https://api.openai.com/v1', 'OPENAI_API_KEY'); \
		INSERT INTO llm_profiles (id, provider_id, name, model, protocol, endpoint_path, timeout_ms) VALUES ('openai-gpt4', 'openai', 'GPT-4.1 Mini', 'gpt-4.1-mini', 'openai_chat_completions', '/chat/completions', 60000); \
		INSERT INTO llm_policies (purpose, profile_chain_json) VALUES ('translation_refined', '[\"openai-gpt4\"]'::jsonb);"

db-seed:
	psql "postgres:///typoon_test?sslmode=disable" -c "\
		INSERT INTO coin_packages (name, xu_amount, bonus_xu, price_vnd, sort_order) VALUES \
		('Gói 100 xu', 100, 0, 10000, 1), \
		('Gói 500 xu', 500, 50, 45000, 2), \
		('Gói 1000 xu', 1000, 150, 85000, 3), \
		('Gói 5000 xu', 5000, 1000, 400000, 4); \
		INSERT INTO users (id, display_name, is_admin) VALUES (gen_random_uuid(), 'Test User', false) ON CONFLICT DO NOTHING; \
		INSERT INTO wallet_accounts (user_id, available_xu, held_xu) SELECT id, 0, 0 FROM users LIMIT 1 ON CONFLICT DO NOTHING; \
		INSERT INTO llm_providers (id, name, kind, base_url, api_key_ref) VALUES ('openai', 'OpenAI', 'openai', 'https://api.openai.com/v1', 'OPENAI_API_KEY') ON CONFLICT DO NOTHING; \
		INSERT INTO llm_profiles (id, provider_id, name, model, protocol, endpoint_path, timeout_ms) VALUES ('openai-gpt4', 'openai', 'GPT-4.1 Mini', 'gpt-4.1-mini', 'openai_chat_completions', '/chat/completions', 60000) ON CONFLICT DO NOTHING; \
		INSERT INTO llm_policies (purpose, profile_chain_json) VALUES ('translation_refined', '[\"openai-gpt4\"]'::jsonb) ON CONFLICT DO NOTHING;"
