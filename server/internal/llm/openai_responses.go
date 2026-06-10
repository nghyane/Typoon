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

type OpenaiResponses struct {
	http *http.Client
}

func NewOpenaiResponses() OpenaiResponses {
	return OpenaiResponses{http: &http.Client{}}
}

func (p OpenaiResponses) Do(ctx context.Context, profile Profile, req TextRequest) (TextResult, error) {
	if profile.Protocol != "openai_responses" {
		return TextResult{}, fmt.Errorf("unsupported protocol: %s", profile.Protocol)
	}

	timeout := time.Duration(profile.Timeout) * time.Millisecond
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	body := map[string]any{
		"model":        profile.Model,
		"instructions": req.System,
		"input": []map[string]any{{
			"role":    "user",
			"content": []map[string]any{{"type": "input_text", "text": req.User}},
		}},
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
		return TextResult{}, fmt.Errorf("responses upstream status %d: %s", res.StatusCode, string(bodyBytes))
	}

	text, usage, err := parseResponsesStream(res.Body)
	if err != nil {
		return TextResult{}, err
	}

	return TextResult{Text: text, Usage: usage}, nil
}
