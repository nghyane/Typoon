package auth

type DiscordUser struct {
	ID       string
	Username string
	Avatar   string
	Email    string
}

type Session struct {
	UserID   string `json:"userId"`
	Username string `json:"username"`
	Avatar   string `json:"avatar"`
}

type Flow struct {
	ID        string
	State     string
	ReturnURL string
}
