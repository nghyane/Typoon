-- name: GetWallet :one
SELECT user_id, available_xu, held_xu
FROM wallet_accounts
WHERE user_id = $1
LIMIT 1;

-- name: LockWallet :one
SELECT user_id, available_xu, held_xu
FROM wallet_accounts
WHERE user_id = $1
FOR UPDATE;

-- name: UpdateWallet :exec
UPDATE wallet_accounts
SET available_xu = $2, held_xu = $3, updated_at = now()
WHERE user_id = $1;

-- name: AppendLedger :exec
INSERT INTO wallet_ledger
  (user_id, kind, available_delta_xu, held_delta_xu,
   balance_available_after, balance_held_after,
   reference_type, reference_id, note)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);

-- name: ListLedger :many
SELECT id, kind, available_delta_xu, held_delta_xu, reference_type, reference_id, note, created_at
FROM wallet_ledger
WHERE user_id = $1
ORDER BY created_at DESC
LIMIT $2;
