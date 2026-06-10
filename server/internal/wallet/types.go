package wallet

type Wallet struct {
	Available int `json:"available"`
	Held      int `json:"held"`
}

type LedgerEntry struct {
	ID            string `json:"id"`
	Kind          string `json:"kind"`
	AvailableDelta int   `json:"availableDelta"`
	HeldDelta     int   `json:"heldDelta"`
	ReferenceType string `json:"referenceType"`
	ReferenceID   string `json:"referenceId"`
	Note          string `json:"note"`
	CreatedAt     string `json:"createdAt"`
}

type ListLedgerInput struct {
	Limit  int
	Cursor string
}

type ListLedgerOutput struct {
	Entries    []LedgerEntry `json:"entries"`
	NextCursor string        `json:"nextCursor"`
}
