package auth

import (
	"net/http"
	"net/url"

	"github.com/go-chi/chi/v5"

	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type Handler struct {
	store   Store
	discord Discord
}

func NewHandler(store Store, discord Discord) Handler {
	return Handler{store: store, discord: discord}
}

func (h Handler) Mount(r chi.Router) {
	r.Get("/api/auth/discord/start", h.start)
	r.Get("/api/auth/discord/callback", h.callback)
	r.Get("/api/auth/session", h.session)
	r.Post("/api/auth/logout", h.logout)
}

func (h Handler) start(w http.ResponseWriter, r *http.Request) {
	returnURL := r.URL.Query().Get("returnTo")
	if returnURL == "" {
		returnURL = "/"
	}

	flow, err := h.store.CreateFlow(r.Context(), returnURL)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	authURL := h.discord.AuthURL(flow.State)
	http.Redirect(w, r, authURL, http.StatusFound)
}

func (h Handler) callback(w http.ResponseWriter, r *http.Request) {
	state := r.URL.Query().Get("state")
	code := r.URL.Query().Get("code")

	if state == "" || code == "" {
		httpx.Error(w, httpx.BadRequest("invalid_callback", "Missing state or code"))
		return
	}

	flow, err := h.store.ValidateFlow(r.Context(), state)
	if err != nil {
		httpx.Error(w, httpx.BadRequest("invalid_state", "Invalid OAuth state"))
		return
	}

	token, err := h.discord.Exchange(r.Context(), code)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	user, err := h.discord.GetUser(token)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	userID, err := h.store.UpsertUser(r.Context(), user)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	_ = h.store.CreateOAuthAccount(r.Context(), userID, "discord", user.ID)

	sessionToken, err := h.store.CreateSession(r.Context(), userID)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     "__Host-typoon-session",
		Value:    sessionToken,
		Path:     "/",
		HttpOnly: true,
		Secure:   true,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   86400 * 30,
	})

	redirectURL := flow.ReturnURL
	if parsed, err := url.Parse(redirectURL); err == nil && parsed.Path == "" {
		redirectURL = "/"
	}

	http.Redirect(w, r, redirectURL, http.StatusFound)
}

func (h Handler) session(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("__Host-typoon-session")
	if err != nil {
		httpx.Error(w, httpx.Unauthorized("unauthorized", "Not authenticated"))
		return
	}

	session, err := h.store.GetSession(r.Context(), cookie.Value)
	if err != nil {
		httpx.Error(w, httpx.Unauthorized("unauthorized", "Session expired"))
		return
	}

	httpx.JSON(w, http.StatusOK, session)
}

func (h Handler) logout(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("__Host-typoon-session")
	if err == nil {
		_ = h.store.DeleteSession(r.Context(), cookie.Value)
	}

	http.SetCookie(w, &http.Cookie{
		Name:     "__Host-typoon-session",
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		Secure:   true,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   -1,
	})

	httpx.JSON(w, http.StatusOK, map[string]bool{"ok": true})
}
