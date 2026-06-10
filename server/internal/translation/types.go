package translation

type RefineInput struct {
	SessionID    string       `json:"sessionId"    validate:"required"`
	SourceLang   string       `json:"sourceLang"`
	TargetLang   string       `json:"targetLang"   validate:"required"`
	Units        []RefineUnit `json:"units"         validate:"required,min=1"`
	ActiveUnitID []string     `json:"activeUnitIds" validate:"required,min=1"`
	ContextBlock string       `json:"contextBlock"`
}

type RefineUnit struct {
	ID        string `json:"id"        validate:"required"`
	PageIndex int    `json:"pageIndex"`
	Order     int    `json:"order"`
	Source    string `json:"sourceText"`
	Draft     string `json:"draftText"`
	Kind      string `json:"kind"      validate:"required"`
	Role      string `json:"role"`
	FitWidth  int    `json:"fitWidth"`
	FitHeight int    `json:"fitHeight"`
	LinesHint int    `json:"linesHint"`
}

type RefineOutput struct {
	SessionID string        `json:"sessionId"`
	Refined   []RefinedUnit `json:"refined"`
	Missing   []string      `json:"missing"`
	Keys      []PromptKey   `json:"promptKeys"`
	Latency   int           `json:"latencyMs"`
}

type RefinedUnit struct {
	ID        string `json:"id"`
	PageIndex int    `json:"pageIndex"`
	Kind      string `json:"kind"`
	Source    string `json:"sourceText"`
	Draft     string `json:"draftText"`
	Target    string `json:"targetText"`
}

type PromptKey struct {
	Key    string `json:"key"`
	UnitID string `json:"unitId"`
}

type PromptUnit struct {
	RefineUnit
	Key string
}
