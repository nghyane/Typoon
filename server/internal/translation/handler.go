package translation

import (
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type Handler struct {
	usecase  Usecase
	sessionDeps SessionDeps
}

func NewHandler(u Usecase, sd SessionDeps) Handler {
	return Handler{usecase: u, sessionDeps: sd}
}

func (h Handler) Mount(r chi.Router) {
	r.Post("/api/translation-sessions", h.createSession)
	r.Post("/api/translation-sessions/{id}/refine-windows", h.refine)
	r.Post("/api/translation-sessions/{id}/finish", h.finish)
	r.Post("/api/translation-sessions/{id}/cancel", h.cancel)
}

func (h Handler) createSession(w http.ResponseWriter, r *http.Request) {
	var input SessionInput
	if err := httpx.Decode(r, &input); err != nil {
		httpx.Error(w, err)
		return
	}

	output, err := h.usecase.CreateSession(r.Context(), h.sessionDeps, input)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusCreated, output)
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

func (h Handler) finish(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	output, err := h.usecase.FinishSession(r.Context(), h.sessionDeps, id)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, output)
}

func (h Handler) cancel(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	output, err := h.usecase.CancelSession(r.Context(), h.sessionDeps, id)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, output)
}
