package translation

import (
	"regexp"
	"strings"
)

var headerRe = regexp.MustCompile(`^@@ ([A-Z0-9]{7}) (dialogue|sfx|skip)\s*$`)

type ParsedBlock struct {
	Key  string
	Kind string
	Text string
}

func Parse(text string, activeKeys map[string]bool) []ParsedBlock {
	var out []ParsedBlock

	seen := make(map[string]bool)

	for _, block := range iterBlocks(text) {
		if !activeKeys[block.Key] || seen[block.Key] {
			continue
		}

		seen[block.Key] = true

		if block.Kind == "skip" {
			out = append(out, ParsedBlock{Key: block.Key, Kind: "skip", Text: ""})

			continue
		}

		if block.Text == "" {
			delete(seen, block.Key)

			continue
		}

		out = append(out, block)
	}

	return out
}

func iterBlocks(text string) []ParsedBlock {
	text = strip(text)

	var out []ParsedBlock
	var current *ParsedBlock

	for _, line := range strings.Split(text, "\n") {
		m := headerRe.FindStringSubmatch(line)
		if m != nil {
			if current != nil {
				out = append(out, *current)
			}

			current = &ParsedBlock{Key: m[1], Kind: m[2]}
		} else if current != nil {
			current.Text = current.Text + "\n" + line
		}
	}

	if current != nil {
		current.Text = strings.TrimSpace(current.Text)
		out = append(out, *current)
	}

	return out
}

func strip(text string) string {
	s := text

	if idx := strings.LastIndex(s, "</think>"); idx != -1 {
		s = s[idx+len("</think>"):]
	}

	s = strings.TrimSpace(s)

	if strings.HasPrefix(s, "```") {
		s = s[strings.IndexByte(s, '\n')+1:]

		if strings.HasSuffix(s, "```") {
			s = s[:len(s)-3]
		}
	}

	return s
}
