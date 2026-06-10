package httpx

import (
	"encoding/json"
	"net/http"

	"github.com/go-playground/validator/v10"
)

var validate = validator.New()

func Decode(r *http.Request, out any) error {
	r.Body = http.MaxBytesReader(nil, r.Body, 1<<20)

	if err := json.NewDecoder(r.Body).Decode(out); err != nil {
		return BadRequest("invalid_request", "Invalid request body")
	}

	if err := validate.Struct(out); err != nil {
		return BadRequest("invalid_request", "Invalid request body")
	}

	return nil
}
