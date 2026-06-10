package payment

import (
	"context"

	"github.com/jackc/pgx/v5/pgxpool"

	sqlc "github.com/nghiahoang/typoon-api/internal/db/sqlc"
)

type Store struct {
	db *pgxpool.Pool
	q  *sqlc.Queries
}

func NewStore(db *pgxpool.Pool) Store {
	return Store{db: db, q: sqlc.New()}
}

func (s Store) CreatePending(ctx context.Context, pkg any) (Order, error) {
	return Order{}, nil
}

func (s Store) Find(ctx context.Context, id string) (Order, error) {
	return Order{}, nil
}

func (s Store) AttachCheckout(ctx context.Context, id, url string) error {
	return nil
}

func (s Store) ApplyPaid(ctx context.Context, orderCode string, amount int) error {
	return nil
}
