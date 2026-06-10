package httpx

import (
	"encoding/json"
	"errors"
	"net/http"
)

type AppError struct {
	Status  int    `json:"-"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e AppError) Error() string { return e.Code }

func BadRequest(code, message string) AppError {
	return AppError{Status: http.StatusBadRequest, Code: code, Message: message}
}

func NotFound(code, message string) AppError {
	return AppError{Status: http.StatusNotFound, Code: code, Message: message}
}

func Conflict(code, message string) AppError {
	return AppError{Status: http.StatusConflict, Code: code, Message: message}
}

func FailedDependency(code, message string) AppError {
	return AppError{Status: http.StatusFailedDependency, Code: code, Message: message}
}

func JSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func Error(w http.ResponseWriter, err error) {
	var appErr AppError
	if errors.As(err, &appErr) {
		JSON(w, appErr.Status, map[string]any{
			"error": map[string]string{
				"code":    appErr.Code,
				"message": appErr.Message,
			},
		})
		return
	}

	JSON(w, http.StatusInternalServerError, map[string]any{
		"error": map[string]string{
			"code":    "internal_server_error",
			"message": "Internal server error",
		},
	})
}
