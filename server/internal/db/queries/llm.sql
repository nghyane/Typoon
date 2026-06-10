-- name: GetProfilesForPurpose :many
SELECT
  p.id, p.provider_id, p.model, p.protocol, p.endpoint_path, p.timeout_ms,
  pr.base_url, pr.api_key_ref
FROM llm_policies policy
CROSS JOIN LATERAL jsonb_array_elements_text(policy.profile_chain_json)
  WITH ORDINALITY AS chain(profile_id, ord)
JOIN llm_profiles p ON p.id = chain.profile_id
JOIN llm_providers pr ON pr.id = p.provider_id
WHERE policy.purpose = $1
  AND policy.enabled = true
  AND p.enabled = true
  AND pr.enabled = true
ORDER BY chain.ord;
