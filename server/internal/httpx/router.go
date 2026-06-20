package httpx

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/cors"
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
	Settings    Module
	Translation Module
}

func NewRouter(deps Deps, allowedOrigins []string) chi.Router {
	r := chi.NewRouter()

	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   allowedOrigins,
		AllowedMethods:   []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

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
	deps.Settings.Mount(r)
	deps.Translation.Mount(r)

	return r
}
