package payment

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
)

type PayOSClient struct {
	clientID    string
	apiKey      string
	checksumKey string
	http        *http.Client
}

type PayOSConfig struct {
	ClientID    string
	APIKey      string
	ChecksumKey string
	ReturnURL   string
	CancelURL   string
}

func NewPayOS(cfg PayOSConfig) PayOSClient {
	return PayOSClient{
		clientID:    cfg.ClientID,
		apiKey:      cfg.APIKey,
		checksumKey: cfg.ChecksumKey,
		http:        &http.Client{},
	}
}

type createPaymentReq struct {
	OrderCode   int64  `json:"orderCode"`
	Amount      int    `json:"amount"`
	Description string `json:"description"`
	Items       []item `json:"items"`
	ReturnURL   string `json:"returnUrl"`
	CancelURL   string `json:"cancelUrl"`
}

type item struct {
	Name     string `json:"name"`
	Quantity int    `json:"quantity"`
	Price    int    `json:"price"`
}

type createPaymentRes struct {
	Code        string `json:"code"`
	Desc        string `json:"desc"`
	CheckoutURL string `json:"checkoutUrl"`
}

func (p PayOSClient) CreateOrder(order Order) (string, error) {
	body := createPaymentReq{
		OrderCode:   mustParseInt(order.ProviderCode),
		Amount:      order.Amount,
		Description: fmt.Sprintf("Nap %d xu", order.Xu),
		Items: []item{{
			Name:     fmt.Sprintf("Goi %d xu", order.Xu),
			Quantity: 1,
			Price:    order.Amount,
		}},
		ReturnURL: "",
		CancelURL: "",
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://api-merchant.payos.vn/v2/payment-requests", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-client-id", p.clientID)
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("x-signature", p.sign(payload))

	res, err := p.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("payos: %w", err)
	}
	defer res.Body.Close()

	var out createPaymentRes
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return "", err
	}

	if out.Code != "00" {
		return "", fmt.Errorf("payos: code=%s desc=%s", out.Code, out.Desc)
	}

	return out.CheckoutURL, nil
}

func (p PayOSClient) VerifyWebhook(body []byte, signature string) bool {
	expected := p.sign(body)
	return hmac.Equal([]byte(expected), []byte(signature))
}

func (p PayOSClient) sign(payload []byte) string {
	mac := hmac.New(sha256.New, []byte(p.checksumKey))
	mac.Write(payload)
	return hex.EncodeToString(mac.Sum(nil))
}

func mustParseInt(s string) int64 {
	var n int64
	fmt.Sscanf(s, "%d", &n)
	return n
}
