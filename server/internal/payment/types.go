package payment

type Order struct {
	ID               string `json:"id"`
	Provider         string `json:"provider"`
	ProviderCode     string `json:"providerOrderCode"`
	CoinPackageID    string `json:"coinPackageId"`
	Amount           int    `json:"amountVnd"`
	Xu               int    `json:"xuAmount"`
	Status           string `json:"status"`
	CheckoutURL      string `json:"checkoutUrl"`
	CreatedAt        string `json:"createdAt"`
	PaidAt           string `json:"paidAt"`
}

type CreateInput struct {
	CoinPackageID  string `json:"coinPackageId"  validate:"required"`
	IdempotencyKey string `json:"idempotencyKey"`
}

type CreateOutput struct {
	Order Order `json:"order"`
}

type GetOutput struct {
	Order Order `json:"order"`
}

type WebhookInput struct {
	Body string `json:"body"`
	Sig  string `json:"sig"`
}
