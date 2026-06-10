package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OpenAIChat struct {
	http *http.Client
}

func NewOpenAIChat() OpenAIChat {
	return OpenAIChat{http: &http.Client{}}
}

func (p OpenAIChat) Do(ctx context.Context, profile Profile, req TextRequest) (TextResult, error) {
	if profile.Protocol != "openai_chat_completions" {
		return TextResult{}, fmt.Errorf("unsupported protocol: %s", profile.Protocol)
	}

	timeout := time.Duration(profile.Timeout) * time.Millisecond
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	body := map[string]any{
		"model": profile.Model,
		"messages": []map[string]string{
			{"role": "system", "content": req.System},
			{"role": "user", "content": req.User},
		},
		"stream": true,
	}

	payload, _ := json.Marshal(body)
	url := joinURL(profile.BaseURL, profile.EndpointPath)

	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+profile.APIKey)

	res, err := p.http.Do(httpReq)
	if err != nil {
		return TextResult{}, err
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(res.Body, 1024))
		return TextResult{}, fmt.Errorf("chat upstream status %d: %s", res.StatusCode, string(bodyBytes))
	}

	text, usage, err := parseChatStream(res.Body)
	if err != nil {
		return TextResult{}, err
	}

	return TextResult{Text: text, Usage: usage}, nil
}

// DoStream calls onDelta for each token chunk from the chat completions stream.
func (p OpenAIChat) DoStream(ctx context.Context, profile Profile, req TextRequest, onDelta func(delta string)) (Usage, error) {
	res, cancel, err := p.request(ctx, profile, req)
	if err != nil {
		return Usage{}, err
	}
	defer cancel()
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(res.Body, 1024))
		return Usage{}, fmt.Errorf("chat upstream status %d: %s", res.StatusCode, string(bodyBytes))
	}

	return streamDeltas(res.Body, parseChatStream, onDelta)
}

func (p OpenAIChat) request(ctx context.Context, profile Profile, req TextRequest) (*http.Response, context.CancelFunc, error) {
	timeout := time.Duration(profile.Timeout) * time.Millisecond
	ctx, cancel := context.WithTimeout(ctx, timeout)

	body := map[string]any{
		"model": profile.Model,
		"messages": []map[string]string{
			{"role": "system", "content": req.System},
			{"role": "user", "content": req.User},
		},
		"stream": true,
	}

	payload, _ := json.Marshal(body)
	url := joinURL(profile.BaseURL, profile.EndpointPath)

	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+profile.APIKey)

	res, err := p.http.Do(httpReq)
	return res, cancel, err
}
