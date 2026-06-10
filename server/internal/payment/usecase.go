package payment

import (
	"context"
	"strconv"
	"time"

	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type Usecase struct {
	Store Store
	Payos PayOSClient
}

func (u Usecase) Create(ctx context.Context, input CreateInput) (CreateOutput, error) {
	if input.CoinPackageID == "" {
		return CreateOutput{}, httpx.BadRequest("coin_package_id_required", "Coin package id is required")
	}

	pkg, err := u.Store.requirePackage(ctx, input.CoinPackageID)
	if err != nil {
		return CreateOutput{}, err
	}

	orderCode := strconv.FormatInt(time.Now().UnixNano(), 10)

	order, err := u.Store.createPending(ctx, pkg, orderCode)
	if err != nil {
		return CreateOutput{}, err
	}

	checkoutURL, err := u.Payos.CreateOrder(order)
	if err != nil {
		return CreateOutput{}, err
	}

	_ = u.Store.attachCheckout(ctx, order.ID, checkoutURL)

	return CreateOutput{Order: order}, nil
}

func (u Usecase) Get(ctx context.Context, id string) (GetOutput, error) {
	if id == "" {
		return GetOutput{}, httpx.BadRequest("payment_order_id_required", "Payment order id is required")
	}

	order, err := u.Store.requireOrder(ctx, id)
	if err != nil {
		return GetOutput{}, err
	}

	return GetOutput{Order: order}, nil
}

func (u Usecase) ReceiveWebhook(ctx context.Context, input WebhookInput) error {
	if input.Sig == "" {
		return httpx.BadRequest("payos_signature_required", "PayOS signature is required")
	}

	if !u.Payos.VerifyWebhook([]byte(input.Body), input.Sig) {
		return httpx.BadRequest("payos_signature_invalid", "PayOS signature is invalid")
	}

	return nil
}
