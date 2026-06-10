package httpx

import (
	"net/http"

	"github.com/go-chi/chi/v5"
)

type Module interface {
	Mount(r chi.Router)
}

type Deps struct {
	Auth        Module
	Coin        Module
	LLM         Module
	Wallet      Module
	Payment     Module
	Translation Module
}

func NewRouter(deps Deps) chi.Router {
	r := chi.NewRouter()

	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		JSON(w, http.StatusOK, map[string]any{
			"ok":      true,
			"service": "typoon-api",
		})
	})

	deps.Auth.Mount(r)
	deps.Coin.Mount(r)
	deps.LLM.Mount(r)
	deps.Wallet.Mount(r)
	deps.Payment.Mount(r)
	deps.Translation.Mount(r)

	return r
}
