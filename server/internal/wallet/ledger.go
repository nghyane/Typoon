package wallet

import (
	"context"
	"fmt"
)

type Ledger struct {
	store Store
}

func NewLedger(store Store) Ledger {
	return Ledger{store: store}
}

func (l Ledger) Hold(ctx context.Context, userID, sessionID string, xu int) error {
	if l.store.db == nil {
		return fmt.Errorf("ledger: db not connected")
	}

	// Transaction:
	// 1. Lock wallet FOR UPDATE
	// 2. Check available >= xu
	// 3. Update available -= xu, held += xu
	// 4. Append wallet_ledger kind=hold
	return nil
}

func (l Ledger) Capture(ctx context.Context, userID, sessionID string, xu int) error {
	if l.store.db == nil {
		return fmt.Errorf("ledger: db not connected")
	}

	// Transaction:
	// 1. Update translation_holds state=captured
	// 2. Update wallet held -= xu
	// 3. Append wallet_ledger kind=capture
	return nil
}

func (l Ledger) Release(ctx context.Context, userID, sessionID string, xu int) error {
	if l.store.db == nil {
		return fmt.Errorf("ledger: db not connected")
	}

	// Transaction:
	// 1. Update translation_holds state=released
	// 2. Update wallet held -= xu, available += xu
	// 3. Append wallet_ledger kind=release
	return nil
}
