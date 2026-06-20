package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	flowTTL    = 10 * time.Minute
	sessionTTL = 30 * 24 * time.Hour
)

var (
	ErrDatabaseNotConfigured = errors.New("auth store: database not configured")
	ErrInvalidFlow           = errors.New("auth flow invalid")
	ErrSessionNotFound       = errors.New("auth session not found")
)

type Store struct {
	db *pgxpool.Pool
}

func NewStore(db *pgxpool.Pool) Store {
	return Store{db: db}
}

func (s Store) CreateFlow(ctx context.Context, returnURL string) (Flow, error) {
	if returnURL == "" {
		returnURL = "/"
	}

	state := randomHex(32)
	db, err := s.dbOrErr()
	if err != nil {
		return Flow{}, err
	}

	body, err := json.Marshal(flowContext{ReturnTo: returnURL})
	if err != nil {
		return Flow{}, err
	}

	var id string
	err = db.QueryRow(ctx, `
		INSERT INTO flows (kind, state_hash, context_json, expires_at)
		VALUES ('auth_discord', $1, $2::jsonb, $3)
		RETURNING id::text
	`, hashSecret(state), string(body), time.Now().Add(flowTTL)).Scan(&id)
	if err != nil {
		return Flow{}, err
	}

	return Flow{ID: id, State: state, ReturnURL: returnURL}, nil
}

func (s Store) ValidateFlow(ctx context.Context, state string) (Flow, error) {
	if state == "" {
		return Flow{}, ErrInvalidFlow
	}

	db, err := s.dbOrErr()
	if err != nil {
		return Flow{}, err
	}

	var id string
	var raw []byte
	err = db.QueryRow(ctx, `
		UPDATE flows
		SET status = 'completed', completed_at = now()
		WHERE id = (
			SELECT id
			FROM flows
			WHERE kind = 'auth_discord'
			  AND state_hash = $1
			  AND status = 'pending'
			  AND expires_at > now()
			LIMIT 1
		)
		RETURNING id::text, context_json
	`, hashSecret(state)).Scan(&id, &raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return Flow{}, ErrInvalidFlow
	}
	if err != nil {
		return Flow{}, err
	}

	var fc flowContext
	if err := json.Unmarshal(raw, &fc); err != nil {
		return Flow{}, err
	}
	if fc.ReturnTo == "" {
		fc.ReturnTo = "/"
	}

	return Flow{ID: id, State: state, ReturnURL: fc.ReturnTo}, nil
}

func (s Store) UpsertDiscordUser(ctx context.Context, user DiscordUser) (string, error) {
	db, err := s.dbOrErr()
	if err != nil {
		return "", err
	}

	tx, err := db.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)

	avatarURL := discordAvatarURL(user)
	var userID string
	err = tx.QueryRow(ctx, `
		SELECT u.id::text
		FROM oauth_accounts oa
		JOIN users u ON u.id = oa.user_id
		WHERE oa.provider = 'discord' AND oa.provider_account_id = $1
	`, user.ID).Scan(&userID)
	switch {
	case err == nil:
		_, err = tx.Exec(ctx, `
			UPDATE users
			SET display_name = $2, avatar_url = $3, updated_at = now()
			WHERE id = $1::uuid
		`, userID, nilIfEmpty(user.Username), nilIfEmpty(avatarURL))
		if err != nil {
			return "", err
		}

		_, err = tx.Exec(ctx, `
			UPDATE oauth_accounts
			SET username = $2, avatar_url = $3, updated_at = now()
			WHERE user_id = $1::uuid AND provider = 'discord'
		`, userID, nilIfEmpty(user.Username), nilIfEmpty(avatarURL))
		if err != nil {
			return "", err
		}

	case errors.Is(err, pgx.ErrNoRows):
		err = tx.QueryRow(ctx, `
			INSERT INTO users (display_name, avatar_url)
			VALUES ($1, $2)
			RETURNING id::text
		`, nilIfEmpty(user.Username), nilIfEmpty(avatarURL)).Scan(&userID)
		if err != nil {
			return "", err
		}

		_, err = tx.Exec(ctx, `
			INSERT INTO oauth_accounts (user_id, provider, provider_account_id, username, avatar_url)
			VALUES ($1::uuid, 'discord', $2, $3, $4)
		`, userID, user.ID, nilIfEmpty(user.Username), nilIfEmpty(avatarURL))
		if err != nil {
			return "", err
		}

	default:
		return "", err
	}

	if err := tx.Commit(ctx); err != nil {
		return "", err
	}

	return userID, nil
}

func (s Store) CreateSession(ctx context.Context, userID string) (string, error) {
	token := randomHex(64)
	db, err := s.dbOrErr()
	if err != nil {
		return "", err
	}

	_, err = db.Exec(ctx, `
		INSERT INTO auth_sessions (user_id, token_hash, origin, expires_at)
		VALUES ($1::uuid, $2, 'web', $3)
	`, userID, hashSecret(token), time.Now().Add(sessionTTL))
	if err != nil {
		return "", err
	}

	return token, nil
}

func (s Store) GetSession(ctx context.Context, token string) (DiscordUser, error) {
	if token == "" {
		return DiscordUser{}, ErrSessionNotFound
	}
	db, err := s.dbOrErr()
	if err != nil {
		return DiscordUser{}, err
	}

	var user DiscordUser
	err = db.QueryRow(ctx, `
		UPDATE auth_sessions s
		SET last_seen_at = now()
		FROM users u
		WHERE s.user_id = u.id
		  AND s.token_hash = $1
		  AND s.revoked_at IS NULL
		  AND s.expires_at > now()
		RETURNING u.id::text, COALESCE(u.display_name, ''), COALESCE(u.avatar_url, ''), u.is_admin
	`, hashSecret(token)).Scan(&user.ID, &user.Username, &user.Avatar, &user.IsAdmin)
	if errors.Is(err, pgx.ErrNoRows) {
		return DiscordUser{}, ErrSessionNotFound
	}
	if err != nil {
		return DiscordUser{}, err
	}

	return user, nil
}

func (s Store) DeleteSession(ctx context.Context, token string) error {
	if token == "" {
		return nil
	}
	db, err := s.dbOrErr()
	if err != nil {
		return err
	}

	_, err = db.Exec(ctx, `
		UPDATE auth_sessions
		SET revoked_at = now()
		WHERE token_hash = $1 AND revoked_at IS NULL
	`, hashSecret(token))
	return err
}

func (s Store) dbOrErr() (*pgxpool.Pool, error) {
	if s.db == nil {
		return nil, ErrDatabaseNotConfigured
	}
	return s.db, nil
}

type flowContext struct {
	ReturnTo string `json:"returnTo"`
}

func nilIfEmpty(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func discordAvatarURL(user DiscordUser) string {
	if user.Avatar == "" || user.ID == "" {
		return ""
	}
	if strings.HasPrefix(user.Avatar, "http://") || strings.HasPrefix(user.Avatar, "https://") {
		return user.Avatar
	}

	ext := "png"
	if strings.HasPrefix(user.Avatar, "a_") {
		ext = "gif"
	}
	return fmt.Sprintf("https://cdn.discordapp.com/avatars/%s/%s.%s", user.ID, user.Avatar, ext)
}

func hashSecret(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
