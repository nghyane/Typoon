-- name: ListEnabledCoinPackages :many
SELECT id, name, xu_amount, bonus_xu, price_vnd
FROM coin_packages
WHERE enabled = true
ORDER BY sort_order ASC, price_vnd ASC;
