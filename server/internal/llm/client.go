package llm

import (
	"context"
	"fmt"
	"time"
)

// ── Non-streaming client (collects entire response) ──────────────

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
		return TextResult{}, fmt.Errorf("llm: no profiles")
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
				ProviderID: profile.ProviderID, ProfileID: profile.ID,
				Status: "success", Latency: int(time.Since(started).Milliseconds()),
			})
			result.ProfileID = profile.ID
			result.ProviderID = profile.ProviderID
			result.Attempts = attempts
			return result, nil
		}

		failure := classify(err)
		lastErr = err
		attempts = append(attempts, Attempt{
			ProviderID: profile.ProviderID, ProfileID: profile.ID,
			Status: "failed", ErrorCode: failure.code,
			Latency: int(time.Since(started).Milliseconds()),
		})
		if !failure.fallbackable {
			break
		}
	}

	return TextResult{}, fmt.Errorf("llm: all profiles failed: %w", lastErr)
}

func (c Client) enabledProfiles() []Profile { return c.profiles }

// ── Streaming client (calls onDelta per token) ───────────────────

type StreamClient struct {
	profiles  []Profile
	protocols map[string]StreamProtocol
}

type StreamProtocol interface {
	DoStream(ctx context.Context, profile Profile, req TextRequest, onDelta func(delta string)) (Usage, error)
}

func NewStreamClient(profiles []Profile, protocols map[string]StreamProtocol) StreamClient {
	return StreamClient{profiles: profiles, protocols: protocols}
}

func (c StreamClient) Generate(ctx context.Context, req TextRequest, onDelta func(delta string)) (Usage, error) {
	if len(c.profiles) == 0 {
		return Usage{}, fmt.Errorf("llm: no profiles")
	}

	profile := c.profiles[0]
	protocol, ok := c.protocols[profile.Protocol]
	if !ok {
		return Usage{}, fmt.Errorf("llm: unsupported protocol %q", profile.Protocol)
	}

	return protocol.DoStream(ctx, profile, req, onDelta)
}

type failureType struct {
	code         string
	fallbackable bool
}

func classify(err error) failureType {
	return failureType{code: "llm_error", fallbackable: false}
}
