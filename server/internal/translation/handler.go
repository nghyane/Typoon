package translation

import (
	"fmt"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/nghiahoang/typoon-api/internal/httpx"
	"github.com/nghiahoang/typoon-api/internal/llm"
)

type Handler struct {
	usecase     Usecase
	sessionDeps SessionDeps
}

func NewHandler(u Usecase, sd SessionDeps) Handler {
	return Handler{usecase: u, sessionDeps: sd}
}

func (h Handler) Mount(r chi.Router) {
	r.Post("/api/translation-sessions", h.createSession)
	r.Post("/api/translation-sessions/{id}/refine-windows", h.refine)
	r.Post("/api/translation-sessions/{id}/refine-windows/stream", h.refineStream)
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

// refineStream sends SSE events as the LLM generates text.
func (h Handler) refineStream(w http.ResponseWriter, r *http.Request) {
	var input RefineInput
	if err := httpx.Decode(r, &input); err != nil {
		httpx.Error(w, err)
		return
	}

	sessionID := chi.URLParam(r, "id")
	if sessionID != input.SessionID {
		httpx.Error(w, httpx.BadRequest("session_id_mismatch", "sessionId mismatch"))
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		httpx.Error(w, fmt.Errorf("streaming not supported"))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	system, err := h.usecase.Prompts.RefineSystem(RefinePromptData{
		SourceLang: langName(input.SourceLang),
		TargetLang: langName(input.TargetLang),
	})
	if err != nil {
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
		flusher.Flush()
		return
	}

	ordered := sortUnits(input.Units)
	usedKeys := make(map[string]bool)
	promptUnits := make([]PromptUnit, 0, len(ordered))
	for _, unit := range ordered {
		key := assignPromptKey(input.SessionID, unit.PageIndex, unit.Order, usedKeys)
		promptUnits = append(promptUnits, PromptUnit{RefineUnit: unit, Key: key})
	}

	user := BuildUserPrompt(input, promptUnits)

	_, err = h.usecase.StreamLLM.Generate(r.Context(), llm.TextRequest{
		System: system,
		User:   user,
	}, func(delta string) {
		fmt.Fprintf(w, "data: %s\n\n", delta)
		flusher.Flush()
	})

	if err != nil {
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
		flusher.Flush()
		return
	}

	fmt.Fprintf(w, "event: done\ndata: {}\n\n")
	flusher.Flush()
}
