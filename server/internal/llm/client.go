package llm

import (
	"context"
	"fmt"
	"time"
)

type Client struct {
	profiles []Profile
	protocol Protocol
}

type Protocol interface {
	Do(ctx context.Context, profile Profile, req TextRequest) (TextResult, error)
}

func NewClient(profiles []Profile, protocol Protocol) Client {
	return Client{profiles: profiles, protocol: protocol}
}

func (c Client) Generate(ctx context.Context, purpose string, req TextRequest) (TextResult, error) {
	enabled := enabledProfiles(c.profiles)
	if len(enabled) == 0 {
		return TextResult{}, fmt.Errorf("llm: no enabled profiles for purpose %q", purpose)
	}

	var attempts []Attempt
	var lastErr error

	for _, profile := range enabled {
		started := time.Now()

		result, err := c.protocol.Do(ctx, profile, req)

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

func enabledProfiles(profiles []Profile) []Profile {
	out := make([]Profile, 0, len(profiles))
	for _, p := range profiles {
		out = append(out, p)
	}
	return out
}
