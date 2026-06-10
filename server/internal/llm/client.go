package llm

import (
	"context"
	"fmt"
	"time"
)

type Client struct {
	profiles  []Profile
	protocols map[string]Protocol
}

type Protocol interface {
	Do(ctx context.Context, profile Profile, req TextRequest) (TextResult, error)
}

func NewClient(profiles []Profile, protocols map[string]Protocol) Client {
	return Client{profiles: profiles, protocols: protocols}
}

func (c Client) Generate(ctx context.Context, purpose string, req TextRequest) (TextResult, error) {
	enabled := c.enabledProfiles()
	if len(enabled) == 0 {
		return TextResult{}, fmt.Errorf("llm: no enabled profiles for purpose %q", purpose)
	}

	var attempts []Attempt
	var lastErr error

	for _, profile := range enabled {
		protocol, ok := c.protocols[profile.Protocol]
		if !ok {
			return TextResult{}, fmt.Errorf("llm: unsupported protocol %q", profile.Protocol)
		}

		started := time.Now()

		result, err := protocol.Do(ctx, profile, req)
		if err == nil {
			attempts = append(attempts, Attempt{
				ProviderID: profile.ProviderID,
				ProfileID:  profile.ID,
				Status:     "success",
				Latency:    int(time.Since(started).Milliseconds()),
			})

			result.ProfileID = profile.ID
			result.ProviderID = profile.ProviderID
			result.Attempts = attempts

			return result, nil
		}

		failure := classify(err)
		lastErr = err

		attempts = append(attempts, Attempt{
			ProviderID: profile.ProviderID,
			ProfileID:  profile.ID,
			Status:     "failed",
			ErrorCode:  failure.code,
			Latency:    int(time.Since(started).Milliseconds()),
		})

		if !failure.fallbackable {
			break
		}
	}

	return TextResult{}, fmt.Errorf("llm: all profiles failed: %w", lastErr)
}

type failureType struct {
	code         string
	fallbackable bool
}

func classify(err error) failureType {
	return failureType{code: "llm_error", fallbackable: false}
}

func (c Client) enabledProfiles() []Profile {
	return c.profiles
}
