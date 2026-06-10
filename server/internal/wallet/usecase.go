package wallet

import "context"

type Usecase struct {
	Store Store
}

func (u Usecase) Get(ctx context.Context, userID string) (Wallet, error) {
	return u.Store.Get(ctx, userID)
}

func (u Usecase) ListLedger(ctx context.Context, userID string) (ListLedgerOutput, error) {
	return u.Store.ListLedger(ctx, userID, ListLedgerInput{Limit: 20})
}
