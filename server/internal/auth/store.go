package auth

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	db *pgxpool.Pool
}

func NewStore(db *pgxpool.Pool) Store {
	return Store{db: db}
}

func (s Store) CreateFlow(ctx context.Context, returnURL string) (Flow, error) {
	state := randomHex(32)
	return Flow{ID: state, State: state, ReturnURL: returnURL}, nil
}

func (s Store) ValidateFlow(ctx context.Context, state string) (Flow, error) {
	if state == "" {
		return Flow{}, fmt.Errorf("invalid state")
	}
	return Flow{ID: state, State: state, ReturnURL: "/"}, nil
}

func (s Store) UpsertUser(ctx context.Context, user DiscordUser) (UserID string, err error) {
	return user.ID, nil
}

func (s Store) CreateOAuthAccount(ctx context.Context, userID, provider, providerAccountID string) error {
	return nil
}

func (s Store) CreateSession(ctx context.Context, userID string) (string, error) {
	return randomHex(64), nil
}

func (s Store) GetSession(ctx context.Context, token string) (DiscordUser, error) {
	return DiscordUser{}, fmt.Errorf("session not found")
}

func (s Store) DeleteSession(ctx context.Context, token string) error {
	return nil
}

type UserID string

func randomHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}
