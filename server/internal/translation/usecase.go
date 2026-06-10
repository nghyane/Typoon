package translation

import (
	"context"

	"github.com/nghiahoang/typoon-api/internal/httpx"
	"github.com/nghiahoang/typoon-api/internal/llm"
)

type Usecase struct {
	LLM       llm.Client
	StreamLLM llm.StreamClient
	Prompts   PromptBook
}

const maxUnitsPerWindow = 100

func (u Usecase) Refine(ctx context.Context, sessionID string, input RefineInput) (RefineOutput, error) {
	if sessionID != input.SessionID {
		return RefineOutput{}, httpx.BadRequest("session_id_mismatch", "sessionId mismatch")
	}

	if len(input.Units) > maxUnitsPerWindow {
		return RefineOutput{}, httpx.BadRequest("refine_too_many_units", "Too many refine units")
	}

	requestedActive := setOf(input.ActiveUnitID)

	if len(requestedActive) == 0 {
		return RefineOutput{}, httpx.BadRequest("active_units_required", "Active unit ids are required")
	}

	orderedUnits := sortUnits(input.Units)
	usedKeys := make(map[string]bool)
	promptUnits := make([]PromptUnit, 0, len(orderedUnits))

	for _, unit := range orderedUnits {
		key := assignPromptKey(input.SessionID, unit.PageIndex, unit.Order, usedKeys)
		promptUnits = append(promptUnits, PromptUnit{RefineUnit: unit, Key: key})
	}

	byKey := make(map[string]PromptUnit, len(promptUnits))
	keyByID := make(map[string]string, len(promptUnits))

	for _, unit := range promptUnits {
		byKey[unit.Key] = unit
		keyByID[unit.ID] = unit.Key
	}

	var autoRefined []RefinedUnit
	activeKeys := make(map[string]bool)

	for id := range requestedActive {
		key, ok := keyByID[id]
		if !ok {
			return RefineOutput{}, httpx.BadRequest("active_unit_not_found", "Active refine unit not found")
		}

		unit := byKey[key]

		if unit.Kind == "skip" || (unit.Source == "" && unit.Draft == "") {
			autoRefined = append(autoRefined, skipUnit(unit))
		} else {
			activeKeys[key] = true
		}
	}

	keys := make([]PromptKey, 0, len(promptUnits))
	for _, unit := range promptUnits {
		keys = append(keys, PromptKey{Key: unit.Key, UnitID: unit.ID})
	}

	if len(activeKeys) == 0 {
		return RefineOutput{
			SessionID: input.SessionID,
			Refined:   autoRefined,
			Missing:   nil,
			Keys:      keys,
			Latency:   0,
		}, nil
	}

	system, err := u.Prompts.RefineSystem(RefinePromptData{
		SourceLang: langName(input.SourceLang),
		TargetLang: langName(input.TargetLang),
	})
	if err != nil {
		return RefineOutput{}, err
	}

	user := BuildUserPrompt(input, promptUnits)

	result, err := u.LLM.Generate(ctx, "translation_refined", llm.TextRequest{
		System: system,
		User:   user,
	})
	if err != nil {
		return RefineOutput{}, err
	}

	refined := make([]RefinedUnit, 0, len(autoRefined)+len(activeKeys))
	refined = append(refined, autoRefined...)

	seen := make(map[string]bool)
	for _, block := range Parse(result.Text, activeKeys) {
		unit, ok := byKey[block.Key]
		if !ok {
			continue
		}

		seen[block.Key] = true
		refined = append(refined, RefinedUnit{
			ID:        unit.ID,
			PageIndex: unit.PageIndex,
			Kind:      block.Kind,
			Source:    unit.Source,
			Draft:     unit.Draft,
			Target:    block.Text,
		})
	}

	var missing []string
	for key := range activeKeys {
		if !seen[key] {
			unit := byKey[key]
			missing = append(missing, unit.ID)
		}
	}

	return RefineOutput{
		SessionID: input.SessionID,
		Refined:   refined,
		Missing:   missing,
		Keys:      keys,
		Latency:   result.Usage.Total,
	}, nil
}

func skipUnit(unit PromptUnit) RefinedUnit {
	return RefinedUnit{
		ID:        unit.ID,
		PageIndex: unit.PageIndex,
		Kind:      "skip",
		Source:    unit.Source,
		Draft:     unit.Draft,
		Target:    "",
	}
}

func sortUnits(units []RefineUnit) []RefineUnit {
	out := make([]RefineUnit, len(units))
	copy(out, units)

	for i := range out {
		for j := i + 1; j < len(out); j++ {
			if out[i].PageIndex > out[j].PageIndex ||
				(out[i].PageIndex == out[j].PageIndex && out[i].Order > out[j].Order) {
				out[i], out[j] = out[j], out[i]
			}
		}
	}

	return out
}
