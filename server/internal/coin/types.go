package coin

type Package struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Xu        int    `json:"xu"`
	Bonus     int    `json:"bonus"`
	Price     int    `json:"price"`
}

type ListResponse struct {
	Packages []Package `json:"packages"`
}
