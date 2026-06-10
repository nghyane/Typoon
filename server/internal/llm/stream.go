package llm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// parseChatStream parses SSE from OpenAI Chat Completions stream.
// Format: data: {"choices":[{"delta":{"content":"..."}}, ...], "usage":...}
func parseChatStream(body io.Reader) (string, Usage, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var text strings.Builder
	var usage Usage

	for scanner.Scan() {
		line := scanner.Text()
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

		if choices, ok := obj["choices"].([]any); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]any); ok {
				if delta, ok := choice["delta"].(map[string]any); ok {
					if content, ok := delta["content"].(string); ok {
						text.WriteString(content)
					}
				}
			}
		}

		if u, ok := obj["usage"].(map[string]any); ok {
			usage = parseUsage(u)
		}
	}

	if err := scanner.Err(); err != nil {
		return "", Usage{}, fmt.Errorf("chat stream: %w", err)
	}

	result := strings.TrimSpace(text.String())
	if result == "" {
		return "", Usage{}, fmt.Errorf("chat stream produced no text")
	}

	return result, usage, nil
}

// parseResponsesStream parses SSE from OpenAI Responses API stream.
// Format: event: response.output_text.delta\n data: {"delta":"..."}
//         event: response.completed\n data: {"response":{"usage":...}}
func parseResponsesStream(body io.Reader) (string, Usage, error) {
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

		var obj map[string]any
		if err := json.Unmarshal([]byte(data), &obj); err != nil {
			continue
		}

		switch event {
		case "response.output_text.delta":
			if delta, ok := obj["delta"].(string); ok {
				text.WriteString(delta)
			}

		case "response.completed":
			if resp, ok := obj["response"].(map[string]any); ok {
				if u, ok := resp["usage"].(map[string]any); ok {
					usage = parseUsage(u)
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return "", Usage{}, fmt.Errorf("responses stream: %w", err)
	}

	result := strings.TrimSpace(text.String())
	if result == "" {
		return "", Usage{}, fmt.Errorf("responses stream produced no text")
	}

	return result, usage, nil
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

func joinURL(base, path string) string {
	if len(base) > 0 && base[len(base)-1] == '/' {
		base = base[:len(base)-1]
	}
	if len(path) > 0 && path[0] == '/' {
		path = path[1:]
	}
	return base + "/" + path
}
