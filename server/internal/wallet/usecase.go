package wallet

import "context"

type Usecase struct {
	Store Store
}

func (u Usecase) Get(ctx context.Context) (Wallet, error) {
	return u.Store.Get(ctx)
}

func (u Usecase) ListLedger(ctx context.Context) (ListLedgerOutput, error) {
	return u.Store.ListLedger(ctx, ListLedgerInput{Limit: 20})
}
