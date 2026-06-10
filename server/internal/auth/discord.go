package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"golang.org/x/oauth2"
)

var discordEndpoint = oauth2.Endpoint{
	AuthURL:  "https://discord.com/oauth2/authorize",
	TokenURL: "https://discord.com/api/oauth2/token",
}

type Discord struct {
	config *oauth2.Config
}

type DiscordConfig struct {
	ClientID     string
	ClientSecret string
	RedirectURL  string
}

func NewDiscord(cfg DiscordConfig) Discord {
	return Discord{
		config: &oauth2.Config{
			ClientID:     cfg.ClientID,
			ClientSecret: cfg.ClientSecret,
			RedirectURL:  cfg.RedirectURL,
			Scopes:       []string{"identify"},
			Endpoint:     discordEndpoint,
		},
	}
}

func (d Discord) AuthURL(state string) string {
	return d.config.AuthCodeURL(state)
}

func (d Discord) Exchange(ctx context.Context, code string) (*oauth2.Token, error) {
	return d.config.Exchange(ctx, code)
}

func (d Discord) GetUser(token *oauth2.Token) (DiscordUser, error) {
	u, err := url.Parse("https://discord.com/api/users/@me")
	if err != nil {
		return DiscordUser{}, err
	}

	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return DiscordUser{}, err
	}

	req.Header.Set("Authorization", "Bearer "+token.AccessToken)

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return DiscordUser{}, fmt.Errorf("discord: %w", err)
	}
	defer res.Body.Close()

	var user DiscordUser
	if err := json.NewDecoder(res.Body).Decode(&user); err != nil {
		return DiscordUser{}, err
	}

	return user, nil
}
