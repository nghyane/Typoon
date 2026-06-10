package llm

type TextRequest struct {
	System string
	User   string
}

type Usage struct {
	Input  int
	Output int
	Total  int
}

type Attempt struct {
	ProviderID string `json:"providerId"`
	ProfileID  string `json:"profileId"`
	Status     string `json:"status"`
	ErrorCode  string `json:"errorCode,omitempty"`
	Latency    int    `json:"latency"`
}

type TextResult struct {
	Text       string    `json:"text"`
	ProfileID  string    `json:"profileId"`
	ProviderID string    `json:"providerId"`
	Usage      Usage     `json:"usage"`
	Attempts   []Attempt `json:"attempts"`
}

type Profile struct {
	ID           string
	ProviderID   string
	Model        string
	Protocol     string
	EndpointPath string
	BaseURL      string
	APIKey       string
	Timeout      int
}

type Config struct {
	Profiles []Profile
}
