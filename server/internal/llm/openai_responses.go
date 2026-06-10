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

// OpenaiResponses calls the OpenAI Responses API (/v1/responses).
// Used by models like gpt-5.4-mini that speak the responses protocol.
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
		OutputText string `json:"output_text"`
		Output     []struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(res.Body).Decode(&decoded); err != nil {
		return TextResult{}, err
	}

	text := decoded.OutputText
	if text == "" {
		for _, item := range decoded.Output {
			for _, block := range item.Content {
				text += block.Text
			}
		}
	}

	if text == "" {
		return TextResult{}, fmt.Errorf("llm responses payload did not include output text")
	}

	return TextResult{
		Text: text,
		Usage: Usage{
			Input:  decoded.Usage.InputTokens,
			Output: decoded.Usage.OutputTokens,
			Total:  decoded.Usage.TotalTokens,
		},
	}, nil
}
