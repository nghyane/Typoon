package llm

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type ProbeHandler struct {
	client Client
}

func NewProbeHandler(client Client) ProbeHandler {
	return ProbeHandler{client: client}
}

func (h ProbeHandler) Mount(r chi.Router) {
	r.Get("/api/llm/probe", h.probe)
}

func (h ProbeHandler) probe(w http.ResponseWriter, r *http.Request) {
	started := time.Now()

	result, err := h.client.Generate(r.Context(), "translation_refined", TextRequest{
		System: "Reply with exactly: OK",
		User:   "Say hello in one word.",
	})

	if err != nil {
		httpx.JSON(w, http.StatusOK, map[string]any{
			"ok":      false,
			"error":   err.Error(),
			"latency": time.Since(started).Milliseconds(),
		})
		return
	}

	httpx.JSON(w, http.StatusOK, map[string]any{
		"ok":      true,
		"text":    result.Text,
		"model":   result.ProfileID,
		"usage":   result.Usage,
		"latency": result.Attempts[len(result.Attempts)-1].Latency,
	})
}
