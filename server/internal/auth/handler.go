package auth

import (
	"errors"
	"net/http"
	"strings"

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
	r.Post("/api/auth/discord/callback", h.callback)
	r.Get("/api/auth/session", h.session)
	r.Post("/api/auth/logout", h.logout)
	r.Post("/api/auth/da/exchange", h.daExchange)
}

func (h Handler) start(w http.ResponseWriter, r *http.Request) {
	returnURL := r.URL.Query().Get("returnTo")
	if returnURL == "" {
		returnURL = "/"
	}

	flow, err := h.store.CreateFlow(r.Context(), returnURL)
	if err != nil {
		writeError(w, err)
		return
	}

	authURL := h.discord.AuthURL(flow.State)
	http.Redirect(w, r, authURL, http.StatusFound)
}

func (h Handler) callback(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Code  string `json:"code"  validate:"required"`
		State string `json:"state" validate:"required"`
	}

	if err := httpx.Decode(r, &input); err != nil {
		httpx.Error(w, err)
		return
	}

	flow, err := h.store.ValidateFlow(r.Context(), input.State)
	if err != nil {
		writeFlowError(w, err)
		return
	}

	token, err := h.discord.Exchange(r.Context(), input.Code)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	user, err := h.discord.GetUser(token)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	userID, err := h.store.UpsertDiscordUser(r.Context(), user)
	if err != nil {
		writeError(w, err)
		return
	}

	sessionToken, err := h.store.CreateSession(r.Context(), userID)
	if err != nil {
		writeError(w, err)
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     "typoon-session",
		Value:    sessionToken,
		Path:     "/",
		HttpOnly: true,
		Secure:   isHTTPS(r),
		SameSite: http.SameSiteLaxMode,
		MaxAge:   86400 * 30,
	})

	returnTo := flow.ReturnURL
	if returnTo == "" {
		returnTo = "/"
	}

	httpx.JSON(w, http.StatusOK, map[string]string{"returnTo": returnTo})
}

func (h Handler) session(w http.ResponseWriter, r *http.Request) {
	sessionToken := cookieToken(r)
	if sessionToken == "" {
		sessionToken = bearerToken(r)
	}

	if sessionToken == "" {
		httpx.Error(w, httpx.Unauthorized("unauthorized", "Not authenticated"))
		return
	}

	user, err := h.store.GetSession(r.Context(), sessionToken)
	if err != nil {
		writeSessionError(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, Session{
		ID:          user.ID,
		DisplayName: user.Username,
		AvatarURL:   user.Avatar,
		IsAdmin:     user.IsAdmin,
	})
}

func cookieToken(r *http.Request) string {
	cookie, err := r.Cookie("typoon-session")
	if err != nil {
		return ""
	}
	return cookie.Value
}

func bearerToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if after, found := strings.CutPrefix(auth, "Bearer "); found {
		return after
	}
	return ""
}

func (h Handler) logout(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("typoon-session")
	if err == nil {
		_ = h.store.DeleteSession(r.Context(), cookie.Value)
	}

	http.SetCookie(w, &http.Cookie{
		Name:     "typoon-session",
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		Secure:   isHTTPS(r),
		SameSite: http.SameSiteLaxMode,
		MaxAge:   -1,
	})

	httpx.JSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// DA exchange — Discord Activity silent auth (no cookie, returns token in body).
// DA iframe calls discordSdk.commands.authorize({ prompt: 'none' }) to get a code,
// then posts it here. Server verifies the code, upserts the user, and returns
// a session token for bearer auth in subsequent requests.
func (h Handler) daExchange(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Code string `json:"code" validate:"required"`
	}

	if err := httpx.Decode(r, &input); err != nil {
		httpx.Error(w, err)
		return
	}

	token, err := h.discord.ExchangeActivity(r.Context(), input.Code)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	user, err := h.discord.GetUser(token)
	if err != nil {
		httpx.Error(w, err)
		return
	}

	userID, err := h.store.UpsertDiscordUser(r.Context(), user)
	if err != nil {
		writeError(w, err)
		return
	}

	sessionToken, err := h.store.CreateSession(r.Context(), userID)
	if err != nil {
		writeError(w, err)
		return
	}

	httpx.JSON(w, http.StatusOK, map[string]string{"token": sessionToken})
}

func writeError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrDatabaseNotConfigured) {
		httpx.Error(w, httpx.FailedDependency("auth_unavailable", "Auth database is not configured"))
		return
	}
	httpx.Error(w, err)
}

func writeFlowError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrInvalidFlow) {
		httpx.Error(w, httpx.BadRequest("invalid_state", "Invalid OAuth state"))
		return
	}
	writeError(w, err)
}

func writeSessionError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrSessionNotFound) {
		httpx.Error(w, httpx.Unauthorized("unauthorized", "Session expired"))
		return
	}
	writeError(w, err)
}

func isHTTPS(r *http.Request) bool {
	return r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https")
}
