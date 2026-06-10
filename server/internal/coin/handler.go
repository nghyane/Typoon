package coin

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
	r.Get("/api/coin-packages", h.list)
}

func (h Handler) list(w http.ResponseWriter, r *http.Request) {
	result, err := h.usecase.List(r.Context())
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, result)
}
