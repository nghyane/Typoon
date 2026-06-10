package coin

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/pgtype"

	sqlc "github.com/nghiahoang/typoon-api/internal/db/sqlc"
)

type Store struct {
	db *pgxpool.Pool
	q  *sqlc.Queries
}

func NewStore(db *pgxpool.Pool) Store {
	return Store{db: db, q: sqlc.New()}
}

func (s Store) List(ctx context.Context) ([]Package, error) {
	if s.db == nil {
		return nil, nil
	}

	rows, err := s.q.ListEnabledCoinPackages(ctx, s.db)
	if err != nil {
		return nil, err
	}

	out := make([]Package, 0, len(rows))

	for _, row := range rows {
		out = append(out, Package{
			ID:    idString(row.ID),
			Name:  row.Name,
			Xu:    int(row.XuAmount),
			Bonus: int(row.BonusXu),
			Price: int(row.PriceVnd),
		})
	}

	return out, nil
}

func idString(uuid pgtype.UUID) string {
	if !uuid.Valid {
		return ""
	}

	return fmt.Sprintf("%x-%x-%x-%x-%x",
		uuid.Bytes[0:4], uuid.Bytes[4:6], uuid.Bytes[6:8], uuid.Bytes[8:10], uuid.Bytes[10:16],
	)
}
