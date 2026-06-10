package wallet

import (
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
	r.Get("/api/wallet", h.wallet)
	r.Get("/api/wallet/ledger", h.ledger)
}

func (h Handler) wallet(w http.ResponseWriter, r *http.Request) {
	result, err := h.usecase.Get(r.Context(), firstUserID(r))
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, result)
}

func (h Handler) ledger(w http.ResponseWriter, r *http.Request) {
	result, err := h.usecase.ListLedger(r.Context(), firstUserID(r))
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, result)
}

func firstUserID(r *http.Request) string {
	userID := r.Header.Get("X-User-ID")
	if userID != "" {
		return userID
	}
	return ""
}
