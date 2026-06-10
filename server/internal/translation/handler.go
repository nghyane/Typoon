package translation

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
	r.Post("/api/translation-sessions/{id}/refine-windows", h.refine)
}

func (h Handler) refine(w http.ResponseWriter, r *http.Request) {
	var input RefineInput

	if err := httpx.Decode(r, &input); err != nil {
		httpx.Error(w, err)
		return
	}

	output, err := h.usecase.Refine(r.Context(), chi.URLParam(r, "id"), input)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, output)
}
