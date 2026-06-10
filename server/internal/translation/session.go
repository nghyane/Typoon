package translation

import (
	"context"

	"github.com/jackc/pgx/v5/pgtype"

	"github.com/nghiahoang/typoon-api/internal/httpx"
)

type SessionInput struct {
	UserID     string `json:"-"`         // from auth middleware
	ContentKey string `json:"contentKey" validate:"required"`
	WorkID     string `json:"workId"`
	SegmentID  string `json:"segmentId"`
	SourceLang string `json:"sourceLang"`
	TargetLang string `json:"targetLang" validate:"required"`
	Mode       string `json:"mode"       validate:"required"`
	PageCount  int    `json:"pageCount"`
	UnitCount  int    `json:"unitCount"`
}

type SessionOutput struct {
	ID          string `json:"id"`
	State       string `json:"state"`
	XuState     string `json:"xuState"`
	PriceXu     int    `json:"priceXu"`
	ContentKey  string `json:"contentKey"`
	TargetLang  string `json:"targetLang"`
	Mode        string `json:"mode"`
	CreatedAt   string `json:"createdAt"`
}

type SessionDeps struct {
	Store  SessionStore
	Ledger interface {
		Hold(ctx context.Context, userID pgtype.UUID, sessionID string, xu int) error
		Capture(ctx context.Context, userID pgtype.UUID, sessionID string, xu int) error
		Release(ctx context.Context, userID pgtype.UUID, sessionID string, xu int) error
	}
}

func (u Usecase) CreateSession(ctx context.Context, deps SessionDeps, input SessionInput) (SessionOutput, error) {
	if input.Mode == "draft_free" {
		return deps.Store.create(ctx, input, "free")
	}

	entitled, _ := deps.Store.hasEntitlement(ctx, input.UserID, input.ContentKey, input.TargetLang)
	if entitled {
		return deps.Store.create(ctx, input, "entitled")
	}

	rule, err := deps.Store.priceRule(ctx, input.Mode)
	if err != nil {
		return SessionOutput{}, httpx.AppError{
			Status:  502,
			Code:    "no_price_rule",
			Message: "No price rule found for mode",
		}
	}

	session, err := deps.Store.create(ctx, input, "held")
	if err != nil {
		return SessionOutput{}, err
	}

	var uid pgtype.UUID
	_ = uid.Scan(input.UserID)

	if err := deps.Ledger.Hold(ctx, uid, session.ID, rule.Xu); err != nil {
		deps.Store.updateState(ctx, session.ID, "error")
		return SessionOutput{}, err
	}

	deps.Store.updateXuState(ctx, session.ID, "held")

	return session, nil
}

type PriceRule struct {
	Xu int
}

type SessionStore struct {
}

func (s SessionStore) create(ctx context.Context, input SessionInput, xuState string) (SessionOutput, error) {
	return SessionOutput{}, nil
}

func (s SessionStore) hasEntitlement(ctx context.Context, userID, contentKey, targetLang string) (bool, error) {
	return false, nil
}

func (s SessionStore) priceRule(ctx context.Context, mode string) (PriceRule, error) {
	return PriceRule{Xu: 0}, nil
}

func (s SessionStore) updateState(ctx context.Context, sessionID string, state string) error {
	return nil
}

func (s SessionStore) updateXuState(ctx context.Context, sessionID string, state string) error {
	return nil
}

func (u Usecase) FinishSession(ctx context.Context, deps SessionDeps, sessionID string) (map[string]string, error) {
	return map[string]string{"status": "done"}, nil
}

func (u Usecase) CancelSession(ctx context.Context, deps SessionDeps, sessionID string) (map[string]string, error) {
	return map[string]string{"status": "cancelled"}, nil
}
