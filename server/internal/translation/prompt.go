package translation

import (
	"bytes"
	_ "embed"
	"fmt"
	"strconv"
	"strings"
	"text/template"
)

//go:embed prompts/refine_system.tmpl
var refineSystemTemplate string

type PromptBook struct {
	refineSystem *template.Template
}

func NewPromptBook() (PromptBook, error) {
	tmpl, err := template.New("refine_system").Parse(refineSystemTemplate)
	if err != nil {
		return PromptBook{}, err
	}

	return PromptBook{refineSystem: tmpl}, nil
}

type RefinePromptData struct {
	SourceLang string
	TargetLang string
}

func (b PromptBook) RefineSystem(data RefinePromptData) (string, error) {
	var buf bytes.Buffer

	if err := b.refineSystem.Execute(&buf, data); err != nil {
		return "", err
	}

	return buf.String(), nil
}

var langNames = map[string]string{
	"ja": "Japanese",
	"ko": "Korean",
	"zh": "Chinese",
	"en": "English",
	"es": "Spanish",
	"vi": "Vietnamese",
}

func langName(code string) string {
	if code == "" {
		return "source language"
	}

	if name, ok := langNames[code]; ok {
		return name
	}

	return code
}

func BuildUserPrompt(input RefineInput, units []PromptUnit) string {
	var b strings.Builder

	if ctx := strings.TrimSpace(input.ContextBlock); ctx != "" {
		b.WriteString("CONTEXT:\n")
		b.WriteString(ctx)
		b.WriteString("\n\n")
	}

	b.WriteString("UNITS:\n")

	activeSet := setOf(input.ActiveUnitID)

	for i, unit := range units {
		if i > 0 {
			b.WriteByte('\n')
		}

		b.WriteString(">>> ")
		b.WriteString(unit.Key)
		b.WriteString(" page=")
		b.WriteString(strconv.Itoa(unit.PageIndex))

		if activeSet[unit.ID] {
			b.WriteString(" active")
		}

		b.WriteString(" kind=")
		b.WriteString(unit.Kind)

		if unit.FitWidth > 0 && unit.FitHeight > 0 {
			lines := unit.LinesHint
			if lines <= 0 {
				lines = max(1, unit.FitHeight/28)
			}

			b.WriteString(fmt.Sprintf(" w=%d h=%d lines=%d", unit.FitWidth, unit.FitHeight, lines))
		}

		b.WriteString("\nSOURCE:\n")
		b.WriteString(strings.TrimSpace(unit.Source))
		b.WriteString("\nDRAFT:\n")
		b.WriteString(strings.TrimSpace(unit.Draft))
	}

	return b.String()
}

func setOf(items []string) map[string]bool {
	m := make(map[string]bool, len(items))
	for _, item := range items {
		m[item] = true
	}

	return m
}
