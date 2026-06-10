package coin

import "context"

type Usecase struct {
	Store Store
}

func (u Usecase) List(ctx context.Context) (ListResponse, error) {
	pkgs, err := u.Store.List(ctx)
	if err != nil {
		return ListResponse{}, err
	}

	return ListResponse{Packages: pkgs}, nil
}
