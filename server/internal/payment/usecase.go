package payment

import (
	"context"
)

type PayOS interface {
	CreateOrder(ctx context.Context, order Order) (string, error)
	VerifyWebhook(body, signature string) (bool, error)
}

type Usecase struct {
	Store Store
	Payos PayOS
}

func (u Usecase) Create(ctx context.Context, input CreateInput) (CreateOutput, error) {
	return CreateOutput{}, nil
}

func (u Usecase) Get(ctx context.Context, id string) (GetOutput, error) {
	return GetOutput{}, nil
}

func (u Usecase) ReceiveWebhook(ctx context.Context, input WebhookInput) error {
	return nil
}
