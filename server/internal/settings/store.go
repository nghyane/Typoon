package settings

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/nghiahoang/typoon-api/internal/auth"
)

var ErrDatabaseNotConfigured = errors.New("settings store: database not configured")

type Store struct {
	db *pgxpool.Pool
}

func NewStore(db *pgxpool.Pool) Store {
	return Store{db: db}
}

func (s Store) Get(ctx context.Context) (Document, error) {
	if s.db == nil {
		return Default(), nil
	}

	var raw []byte
	err := s.db.QueryRow(ctx, `SELECT value FROM app_settings WHERE id = true`).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return Default(), nil
	}
	if err != nil {
		return Document{}, err
	}

	doc := Default()
	if err := json.Unmarshal(raw, &doc); err != nil {
		return Document{}, err
	}
	normalize(&doc)
	return doc, nil
}

func (s Store) Put(ctx context.Context, doc Document, updatedBy string) (Document, error) {
	if s.db == nil {
		return Document{}, ErrDatabaseNotConfigured
	}
	normalize(&doc)

	raw, err := json.Marshal(doc)
	if err != nil {
		return Document{}, err
	}

	_, err = s.db.Exec(ctx, `
		INSERT INTO app_settings (id, value, updated_by)
		VALUES (true, $1::jsonb, $2::uuid)
		ON CONFLICT (id) DO UPDATE
		SET value = EXCLUDED.value,
		    updated_by = EXCLUDED.updated_by,
		    updated_at = now()
	`, string(raw), updatedBy)
	if err != nil {
		return Document{}, err
	}
	return doc, nil
}

func normalize(doc *Document) {
	if len(doc.SourceFetch.Origins) == 0 {
		doc.SourceFetch.Origins = Default().SourceFetch.Origins
	}
	if doc.Pricing.Translation.XuPerPage < 0 {
		doc.Pricing.Translation.XuPerPage = 0
	}
}

type AuthStore interface {
	GetSession(ctx context.Context, token string) (auth.DiscordUser, error)
}
