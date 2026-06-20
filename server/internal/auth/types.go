package auth

type DiscordUser struct {
	ID       string
	Username string
	Avatar   string
	Email    string
	IsAdmin  bool
}

type Session struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	AvatarURL   string `json:"avatar_url"`
	IsAdmin     bool   `json:"is_admin"`
}

type Flow struct {
	ID        string
	State     string
	ReturnURL string
}
