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
		"stream": false,
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return TextResult{}, err
	}

	url := joinURL(profile.BaseURL, profile.EndpointPath)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return TextResult{}, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+profile.APIKey)

	res, err := p.http.Do(httpReq)
	if err != nil {
		return TextResult{}, err
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(res.Body, 1024))
		return TextResult{}, fmt.Errorf("llm upstream status %d: %s", res.StatusCode, string(bodyBytes))
	}

	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(res.Body).Decode(&decoded); err != nil {
		return TextResult{}, err
	}

	if len(decoded.Choices) == 0 {
		return TextResult{}, fmt.Errorf("llm response has no choices")
	}

	return TextResult{
		Text: decoded.Choices[0].Message.Content,
		Usage: Usage{
			Input:  decoded.Usage.PromptTokens,
			Output: decoded.Usage.CompletionTokens,
			Total:  decoded.Usage.TotalTokens,
		},
	}, nil
}

func joinURL(base, path string) string {
	if len(base) > 0 && base[len(base)-1] == '/' {
		base = base[:len(base)-1]
	}

	if len(path) > 0 && path[0] == '/' {
		path = path[1:]
	}

	return base + "/" + path
}
