package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/nghiahoang/typoon-api/internal/coin"
	"github.com/nghiahoang/typoon-api/internal/config"
	"github.com/nghiahoang/typoon-api/internal/httpx"
	"github.com/nghiahoang/typoon-api/internal/llm"
	"github.com/nghiahoang/typoon-api/internal/payment"
	"github.com/nghiahoang/typoon-api/internal/translation"
	"github.com/nghiahoang/typoon-api/internal/wallet"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	ctx := context.Background()

	pool := connectOrNil(ctx, cfg.DatabaseURL)
	if pool != nil {
		defer pool.Close()
	}

	coinStore := coin.NewStore(pool)
	walletStore := wallet.NewStore(pool)
	paymentStore := payment.NewStore(pool)

	prompts, err := translation.NewPromptBook()
	if err != nil {
		log.Fatalf("prompts: %v", err)
	}

	llmClient := llm.NewClient(
		llmConfigFromEnv(),
		llm.NewOpenAIChat(),
	)

	router := httpx.NewRouter(httpx.Deps{
		Coin: coin.NewHandler(coin.Usecase{
			Store: coinStore,
		}),
		Wallet: wallet.NewHandler(wallet.Usecase{
			Store: walletStore,
		}),
		Payment: payment.NewHandler(payment.Usecase{
			Store: paymentStore,
			Payos: payment.NewPayOS(payment.PayOSConfig{
				ClientID:    os.Getenv("PAYOS_CLIENT_ID"),
				APIKey:      os.Getenv("PAYOS_API_KEY"),
				ChecksumKey: os.Getenv("PAYOS_CHECKSUM_KEY"),
			}),
		}),
		Translation: translation.NewHandler(translation.Usecase{
			LLM:     llmClient,
			Prompts: prompts,
		}),
	})

	addr := fmt.Sprintf(":%d", cfg.Port)
	log.Printf("typoon-api listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, router))
}

func connectOrNil(ctx context.Context, url string) *pgxpool.Pool {
	if url == "" {
		log.Println("DATABASE_URL is empty, running without database")
		return nil
	}

	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		log.Fatalf("db: %v", err)
	}

	return pool
}

func llmConfigFromEnv() []llm.Profile {
	return []llm.Profile{
		{
			ID:           "primary",
			ProviderID:   "openai",
			Model:        envDefault("LLM_MODEL", "gpt-4.1-mini"),
			Protocol:     "openai_chat_completions",
			BaseURL:      envDefault("LLM_BASE_URL", "https://api.openai.com/v1"),
			EndpointPath: "/chat/completions",
			APIKey:       os.Getenv("LLM_API_KEY"),
			Timeout:      60_000,
		},
	}
}

func envDefault(key, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}

	return v
}
