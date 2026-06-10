-- name: GetPaymentOrder :one
SELECT id, provider, provider_order_code, coin_package_id, amount_vnd, xu_amount, status, checkout_url, created_at, paid_at
FROM payment_orders
WHERE id = $1
LIMIT 1;

-- name: LockPaymentOrder :one
SELECT id, status, xu_amount, user_id, coin_package_id
FROM payment_orders
WHERE provider_order_code = $1
FOR UPDATE;

-- name: MarkOrderPaid :exec
UPDATE payment_orders
SET status = 'paid', paid_at = now(), updated_at = now()
WHERE id = $1;

-- name: CreatePendingOrder :exec
INSERT INTO payment_orders
  (id, user_id, provider, provider_order_code, coin_package_id, amount_vnd, xu_amount, status, checkout_url)
VALUES ($1, $2, 'payos', $3, $4, $5, $6, 'pending', $7);

-- name: AttachCheckoutURL :exec
UPDATE payment_orders
SET checkout_url = $2, updated_at = now()
WHERE id = $1;
