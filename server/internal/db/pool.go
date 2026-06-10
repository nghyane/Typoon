package db

import (
	"context"

	"github.com/jackc/pgx/v5/pgxpool"
)

func NewPool(ctx context.Context, datbaseURL string) (*pgxpool.Pool, error) {
	return pgxpool.New(ctx, datbaseURL)
}
