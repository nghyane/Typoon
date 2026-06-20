package settings

import (
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type Handler struct {
	store Store
	auth  AuthStore
}

func NewHandler(store Store, auth AuthStore) Handler {
	return Handler{store: store, auth: auth}
}

func (h Handler) Mount(r chi.Router) {
	r.Get("/api/settings", h.publicSettings)
	r.Get("/api/admin/settings", h.adminSettings)
	r.Put("/api/admin/settings", h.putAdminSettings)
}

func (h Handler) publicSettings(w http.ResponseWriter, r *http.Request) {
	doc, err := h.store.Get(r.Context())
	if err != nil {
		httpx.Error(w, err)
		return
	}
	httpx.JSON(w, http.StatusOK, Public(doc))
}

func (h Handler) adminSettings(w http.ResponseWriter, r *http.Request) {
	if _, ok := h.requireAdmin(w, r); !ok {
		return
	}
	doc, err := h.store.Get(r.Context())
	if err != nil {
		httpx.Error(w, err)
		return
	}
	httpx.JSON(w, http.StatusOK, doc)
}

func (h Handler) putAdminSettings(w http.ResponseWriter, r *http.Request) {
	userID, ok := h.requireAdmin(w, r)
	if !ok {
		return
	}

	var doc Document
	if err := httpx.Decode(r, &doc); err != nil {
		httpx.Error(w, err)
		return
	}
	if err := validate(doc); err != nil {
		httpx.Error(w, err)
		return
	}

	saved, err := h.store.Put(r.Context(), doc, userID)
	if err != nil {
		if errors.Is(err, ErrDatabaseNotConfigured) {
			httpx.Error(w, httpx.FailedDependency("settings_unavailable", "Settings database is not configured"))
			return
		}
		httpx.Error(w, err)
		return
	}
	httpx.JSON(w, http.StatusOK, saved)
}

func (h Handler) requireAdmin(w http.ResponseWriter, r *http.Request) (string, bool) {
	user, err := h.auth.GetSession(r.Context(), requestToken(r))
	if err != nil {
		httpx.Error(w, httpx.Unauthorized("unauthorized", "Not authenticated"))
		return "", false
	}
	if !user.IsAdmin {
		httpx.Error(w, httpx.AppError{Status: http.StatusForbidden, Code: "forbidden", Message: "Admin required"})
		return "", false
	}
	return user.ID, true
}

func requestToken(r *http.Request) string {
	if cookie, err := r.Cookie("typoon-session"); err == nil {
		return cookie.Value
	}
	if after, found := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer "); found {
		return after
	}
	return ""
}

func validate(doc Document) error {
	for _, origin := range doc.SourceFetch.Origins {
		if !isOriginURL(origin) {
			return httpx.BadRequest("invalid_settings", "sourceFetch gateway origin must look like https://gateway.example.com")
		}
	}
	return nil
}

func isOriginURL(value string) bool {
	parsed, err := url.Parse(value)
	if err != nil {
		return false
	}
	return (parsed.Scheme == "https" || parsed.Scheme == "http") &&
		parsed.Host != "" &&
		parsed.Path == "" &&
		parsed.RawQuery == "" &&
		parsed.Fragment == ""
}
