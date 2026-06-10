package payment

import (
	"io"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type Handler struct {
	usecase Usecase
}

func NewHandler(u Usecase) Handler {
	return Handler{usecase: u}
}

func (h Handler) Mount(r chi.Router) {
	r.Post("/api/payment-orders", h.create)
	r.Get("/api/payment-orders/{id}", h.get)
	r.Post("/api/payment-webhooks/payos", h.webhook)
}

func (h Handler) create(w http.ResponseWriter, r *http.Request) {
	var input CreateInput
	if err := httpx.Decode(r, &input); err != nil {
		httpx.Error(w, err)
		return
	}

	output, err := h.usecase.Create(r.Context(), input)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusCreated, output)
}

func (h Handler) get(w http.ResponseWriter, r *http.Request) {
	output, err := h.usecase.Get(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, output)
}

func (h Handler) webhook(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(http.MaxBytesReader(nil, r.Body, 1<<16))

	err := h.usecase.ReceiveWebhook(r.Context(), WebhookInput{
		Body: string(body),
		Sig:  r.Header.Get("x-payos-signature"),
	})
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusAccepted, map[string]bool{"accepted": true})
}
