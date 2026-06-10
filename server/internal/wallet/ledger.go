package wallet

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/pgtype"

	sqlc "github.com/nghiahoang/typoon-api/internal/db/sqlc"
	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type Ledger struct {
	pool *pgxpool.Pool
	q    *sqlc.Queries
}

func NewLedger(pool *pgxpool.Pool) Ledger {
	return Ledger{pool: pool, q: sqlc.New()}
}

func (l Ledger) Topup(ctx context.Context, userID pgtype.UUID, orderID string, xu int, vnd int) error {
	return l.withTx(ctx, func(tx sqlc.DBTX) error {
		wallet, err := l.q.LockWallet(ctx, tx, userID)
		if err != nil {
			return fmt.Errorf("ledger topup: %w", err)
		}

		nextAvail := wallet.AvailableXu + int32(xu)

		if err := l.q.UpdateWallet(ctx, tx, &sqlc.UpdateWalletParams{
			UserID:       userID,
			AvailableXu: nextAvail,
			HeldXu:      wallet.HeldXu,
		}); err != nil {
			return err
		}

		note := fmt.Sprintf("Nạp %d xu qua PayOS (%dđ)", xu, vnd)

		return l.q.AppendLedger(ctx, tx, &sqlc.AppendLedgerParams{
			UserID:                userID,
			Kind:                  "topup",
			AvailableDeltaXu:      int32(xu),
			HeldDeltaXu:           0,
			BalanceAvailableAfter: nextAvail,
			BalanceHeldAfter:      wallet.HeldXu,
			ReferenceType:         "payment_order",
			ReferenceID:           orderID,
			Note:                  &note,
		})
	})
}

func (l Ledger) Hold(ctx context.Context, userID pgtype.UUID, sessionID string, xu int) error {
	return l.withTx(ctx, func(tx sqlc.DBTX) error {
		wallet, err := l.q.LockWallet(ctx, tx, userID)
		if err != nil {
			return fmt.Errorf("ledger hold: %w", err)
		}

		if wallet.AvailableXu < int32(xu) {
			return httpx.Conflict("not_enough_xu", "Không đủ xu")
		}

		nextAvail := wallet.AvailableXu - int32(xu)
		nextHeld := wallet.HeldXu + int32(xu)

		if err := l.q.UpdateWallet(ctx, tx, &sqlc.UpdateWalletParams{
			UserID:       userID,
			AvailableXu: nextAvail,
			HeldXu:      nextHeld,
		}); err != nil {
			return err
		}

		return l.q.AppendLedger(ctx, tx, &sqlc.AppendLedgerParams{
			UserID:                userID,
			Kind:                  "hold",
			AvailableDeltaXu:      -int32(xu),
			HeldDeltaXu:           int32(xu),
			BalanceAvailableAfter: nextAvail,
			BalanceHeldAfter:      nextHeld,
			ReferenceType:         "translation_session",
			ReferenceID:           sessionID,
		})
	})
}

func (l Ledger) Capture(ctx context.Context, userID pgtype.UUID, sessionID string, xu int) error {
	return l.withTx(ctx, func(tx sqlc.DBTX) error {
		wallet, err := l.q.LockWallet(ctx, tx, userID)
		if err != nil {
			return fmt.Errorf("ledger capture: %w", err)
		}

		nextHeld := wallet.HeldXu - int32(xu)

		if err := l.q.UpdateWallet(ctx, tx, &sqlc.UpdateWalletParams{
			UserID:       userID,
			AvailableXu: wallet.AvailableXu,
			HeldXu:      nextHeld,
		}); err != nil {
			return err
		}

		return l.q.AppendLedger(ctx, tx, &sqlc.AppendLedgerParams{
			UserID:                userID,
			Kind:                  "capture",
			AvailableDeltaXu:      0,
			HeldDeltaXu:           -int32(xu),
			BalanceAvailableAfter: wallet.AvailableXu,
			BalanceHeldAfter:      nextHeld,
			ReferenceType:         "translation_session",
			ReferenceID:           sessionID,
		})
	})
}

func (l Ledger) Release(ctx context.Context, userID pgtype.UUID, sessionID string, xu int) error {
	return l.withTx(ctx, func(tx sqlc.DBTX) error {
		wallet, err := l.q.LockWallet(ctx, tx, userID)
		if err != nil {
			return fmt.Errorf("ledger release: %w", err)
		}

		nextAvail := wallet.AvailableXu + int32(xu)
		nextHeld := wallet.HeldXu - int32(xu)

		if err := l.q.UpdateWallet(ctx, tx, &sqlc.UpdateWalletParams{
			UserID:       userID,
			AvailableXu: nextAvail,
			HeldXu:      nextHeld,
		}); err != nil {
			return err
		}

		return l.q.AppendLedger(ctx, tx, &sqlc.AppendLedgerParams{
			UserID:                userID,
			Kind:                  "release",
			AvailableDeltaXu:      int32(xu),
			HeldDeltaXu:           -int32(xu),
			BalanceAvailableAfter: nextAvail,
			BalanceHeldAfter:      nextHeld,
			ReferenceType:         "translation_session",
			ReferenceID:           sessionID,
		})
	})
}

func (l Ledger) withTx(ctx context.Context, fn func(tx sqlc.DBTX) error) error {
	if l.pool == nil {
		return fmt.Errorf("ledger: no database connection")
	}

	tx, err := l.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("ledger: %w", err)
	}
	defer tx.Rollback(ctx)

	if err := fn(tx); err != nil {
		return err
	}

	return tx.Commit(ctx)
}
