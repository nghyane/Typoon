package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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
		return TextResult{}, fmt.Errorf("llm upstream status %d: %s", res.StatusCode, string(bodyBytes))
	}

	return parseStream(res.Body)
}

func parseStream(body io.Reader) (TextResult, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var text strings.Builder
	var usage Usage
	var event string

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			event = ""
			continue
		}

		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}

		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		var obj map[string]any
		if err := json.Unmarshal([]byte(data), &obj); err != nil {
			continue
		}

		// Responses API: output_text.delta
		if event == "response.output_text.delta" {
			if delta, ok := obj["delta"].(string); ok {
				text.WriteString(delta)
			}
			continue
		}

		// Responses API: completed
		if event == "response.completed" {
			if resp, ok := obj["response"].(map[string]any); ok {
				if u, ok := resp["usage"].(map[string]any); ok {
					usage = parseUsage(u)
				}
			}
			continue
		}

		// Chat Completions: choices[0].delta.content
		if choices, ok := obj["choices"].([]any); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]any); ok {
				if delta, ok := choice["delta"].(map[string]any); ok {
					if content, ok := delta["content"].(string); ok {
						text.WriteString(content)
					}
				}
			}
		}

		// Usage in chat completions
		if u, ok := obj["usage"].(map[string]any); ok {
			usage = parseUsage(u)
		}
	}

	if err := scanner.Err(); err != nil {
		return TextResult{}, fmt.Errorf("llm stream: %w", err)
	}

	resultText := strings.TrimSpace(text.String())
	if resultText == "" {
		return TextResult{}, fmt.Errorf("llm responses payload did not include output text")
	}

	return TextResult{Text: resultText, Usage: usage}, nil
}

func parseUsage(u map[string]any) Usage {
	return Usage{
		Input:  firstInt(u, "input_tokens", "prompt_tokens"),
		Output: firstInt(u, "output_tokens", "completion_tokens"),
		Total:  firstInt(u, "total_tokens"),
	}
}

func firstInt(m map[string]any, keys ...string) int {
	for _, k := range keys {
		if v, ok := m[k].(float64); ok {
			return int(v)
		}
	}
	return 0
}
