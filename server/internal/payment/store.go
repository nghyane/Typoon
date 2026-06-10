package payment

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	sqlc "github.com/nghiahoang/typoon-api/internal/db/sqlc"
	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type Store struct {
	db *pgxpool.Pool
	q  *sqlc.Queries
}

func NewStore(db *pgxpool.Pool) Store {
	return Store{db: db, q: sqlc.New()}
}

func (s Store) requirePackage(ctx context.Context, id string) (PackageInfo, error) {
	return PackageInfo{}, fmt.Errorf("db not connected")
}

type PackageInfo struct {
	ID    string
	Name  string
	Xu    int
	Price int
}

func (s Store) createPending(ctx context.Context, pkg PackageInfo, orderCode string) (Order, error) {
	return Order{}, fmt.Errorf("db not connected")
}

func (s Store) attachCheckout(ctx context.Context, orderID, url string) error {
	return fmt.Errorf("db not connected")
}

func (s Store) requireOrder(ctx context.Context, id string) (Order, error) {
	return Order{}, httpx.NotFound("payment_order_not_found", "Payment order not found")
}
