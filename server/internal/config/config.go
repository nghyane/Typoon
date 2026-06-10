package config

import "github.com/caarlos0/env/v11"

type Config struct {
	Port        int    `env:"PORT" envDefault:"3000"`
	DatabaseURL string `env:"DATABASE_URL"`
}

func Load() (Config, error) {
	var cfg Config

	err := env.Parse(&cfg)

	return cfg, err
}
