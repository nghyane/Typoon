package wallet

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

func (s Store) Get(ctx context.Context) (Wallet, error) {
	return Wallet{Available: 0, Held: 0}, nil
}

func (s Store) ListLedger(ctx context.Context, input ListLedgerInput) (ListLedgerOutput, error) {
	return ListLedgerOutput{Entries: nil}, nil
}
