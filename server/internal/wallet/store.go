package wallet

import (
	"context"
	"time"

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

func (s Store) Get(ctx context.Context, userID string) (Wallet, error) {
	if s.db == nil {
		return Wallet{}, nil
	}

	uid, err := parseUUID(userID)
	if err != nil {
		return Wallet{}, nil
	}

	row, err := s.q.GetWallet(ctx, s.db, uid)
	if err != nil {
		return Wallet{}, nil
	}

	return Wallet{
		Available: int(row.AvailableXu),
		Held:      int(row.HeldXu),
	}, nil
}

func (s Store) ListLedger(ctx context.Context, userID string, input ListLedgerInput) (ListLedgerOutput, error) {
	if s.db == nil {
		return ListLedgerOutput{}, nil
	}

	uid, err := parseUUID(userID)
	if err != nil {
		return ListLedgerOutput{}, nil
	}

	rows, err := s.q.ListLedger(ctx, s.db, &sqlc.ListLedgerParams{
		UserID: uid,
		Limit:  int32(input.Limit),
	})
	if err != nil {
		return ListLedgerOutput{}, err
	}

	entries := make([]LedgerEntry, 0, len(rows))
	for _, row := range rows {
		entries = append(entries, LedgerEntry{
			Kind:          row.Kind,
			AvailableDelta: int(row.AvailableDeltaXu),
			HeldDelta:      int(row.HeldDeltaXu),
			ReferenceType: row.ReferenceType,
			ReferenceID:   row.ReferenceID,
			Note:          stringPtr(row.Note),
			CreatedAt:     row.CreatedAt.Time.Format(time.RFC3339),
		})
	}

	return ListLedgerOutput{Entries: entries}, nil
}

func parseUUID(s string) (pgtype.UUID, error) {
	var uid pgtype.UUID
	if err := uid.Scan(s); err != nil {
		return uid, err
	}
	return uid, nil
}

func stringPtr(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
