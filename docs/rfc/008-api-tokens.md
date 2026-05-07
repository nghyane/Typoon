# RFC-008: API tokens for tool clients

Status: **implemented** (RFC-008 cutover complete on `main`).
Scope: backend infrastructure để các tool client-side (extension, CLI sau, bot
sau) authenticate vào engine không cần Discord OAuth flow.

## Summary

Thêm long-lived API token bên cạnh JWT bearer hiện tại. User vào Settings tạo
token, copy 1 lần, paste vào extension/CLI. Token chỉ cấp quyền owner (CRUD
project user sở hữu, upload chapter), không có admin. Admin operations vẫn
chỉ qua web SPA + Discord role.

## Why

- **Extension cần long-lived auth**: 30-day JWT đòi user re-login mỗi tháng,
  break tự động. Token revocable theo ý user, hết hạn duy nhất khi user xoá.
- **Engine không phải pull → tool client-side phải POST upload**: cần auth
  channel cho non-browser-OAuth client.
- **Tách quyền**: token = tool, không bao giờ admin. Giảm blast radius nếu
  token leak (không phải là không có hệ luỵ — vẫn upload được vào project user
  sở hữu — nhưng không thể xoá user khác, không thể truy cập project khác,
  không phải admin).

## Non-goals

- Per-token scope (đọc-only, upload-only, …). Phase 1 token là full owner
  permission. Scope sau khi cần.
- Token theo IP / per-host binding.
- OAuth client credentials, refresh token, token rotation tự động.
- Public token (anonymous read).

## Schema

```sql
CREATE TABLE api_tokens (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,                 -- "Chrome ext / Macbook"
    token_hash  TEXT NOT NULL UNIQUE,          -- bcrypt(plaintext)
    prefix      TEXT NOT NULL,                 -- 8 chars đầu, hiển thị UI
    last_used   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at  TIMESTAMPTZ
);
CREATE INDEX idx_api_tokens_user
    ON api_tokens(user_id) WHERE revoked_at IS NULL;
```

Bump `SCHEMA_VERSION` từ `"2"` → `"3"`.

## Token format

```
typ_<32 url-safe random chars>
```

- `typ_` prefix → identify trong logs/UI/Authorization header.
- 32 chars base62 ≈ 190 bits entropy. Đủ.
- Plaintext show **một lần** ở response của `POST /me/tokens`. Mất là tạo
  cái mới.

Hash: bcrypt(plaintext, work_factor=10). Phase 1 community ~1k user, hash
verify mỗi request acceptable. Optimize sau với Redis cache nếu profile chỉ
ra bottleneck.

## Endpoints

```
POST   /api/me/tokens         body: { name }
                              → 201 { id, name, prefix, token: "typ_..." }
GET    /api/me/tokens         → list { id, name, prefix, last_used, created_at }
                                       (KHÔNG plaintext)
DELETE /api/me/tokens/{id}    → 204, set revoked_at
```

`require_user` accept 2 format Authorization:

```python
async def require_user(authorization, db, cfg):
    raw = authorization[7:]  # strip "Bearer "
    if raw.startswith("typ_"):
        user = await db.user_by_api_token(raw)   # lookup by hash, update last_used
        if not user:
            raise HTTPException(401)
        # API token KHÔNG carry roles → admin operations sẽ bị 403 ở layer trên.
        user["roles"] = []
        return user
    # else: JWT path như hiện tại
```

`db.user_by_api_token`:

1. Hash chunk: tìm tất cả token chưa revoke có cùng prefix → bcrypt verify từng
   cái. Prefix cô lập ~99.99% cases (8 chars random ≈ 47 bits → collision rất
   hiếm), thường chỉ verify 1 hash.
2. Update `last_used = NOW()` async (không block response).
3. Return user.

## Auth dependency tree (sau RFC-008)

```
require_user        accepts JWT or API token
require_admin       requires role_id from JWT (API token never admin)
require_project_*   reads user["id"] from require_user — token đủ
```

→ API token thay JWT seamlessly cho mọi route owner-scoped. Không thay được
admin route.

## UI changes

**Settings page** thêm tab "API tokens":

```
┌─ Tokens ─────────────────────────────┐
│ Tokens cho extension/script kết nối  │
│ Typoon. Mỗi tool 1 token.            │
│                                      │
│ Token hiện tại:                      │
│   ┌────────────────────────────────┐ │
│   │ Chrome extension               │ │
│   │ typ_AbCd…    last used 2h ago  │ │
│   │                       [Revoke] │ │
│   └────────────────────────────────┘ │
│                                      │
│ [+ Tạo token mới]                    │
└──────────────────────────────────────┘
```

Modal create:
```
Tên (chỉ bạn thấy, để biết token này dùng đâu):
[Chrome extension              ]

[Huỷ]  [Tạo]

→ sau khi tạo:
  ┌─────────────────────────────────────┐
  │ Đây là lần duy nhất token hiển thị. │
  │ Copy ngay và lưu chỗ an toàn.       │
  │                                     │
  │ typ_AbCdEfGh123…789                 │
  │                          [Copy]     │
  │                                     │
  │ [Đã lưu]                            │
  └─────────────────────────────────────┘
```

## Risks

- **Token leak**: token full owner. Nếu user paste nhầm vào public repo →
  attacker upload spam vào project user. Mitigation: revoke nhanh, audit log
  `last_used` để user tự nhận.
- **Bcrypt verify cost**: profile khi >100 req/s. Phase 1 không lo.
- **Prefix collision**: 8 chars, 62^8 ≈ 218 trillion. 1k user × 5 token = 5k
  prefix → P(collision) ≈ 0. Không cần xử lý.
- **last_used update gây contention**: mỗi request UPDATE 1 row → fine với pg.
  Nếu thấy load tăng, batch update ở SSE timer 60s.

## Order of work

1. Schema + `SCHEMA_VERSION` bump.
2. Storage: `create_api_token`, `user_by_api_token`, `list_api_tokens`,
   `revoke_api_token`.
3. Endpoints `/api/me/tokens` (POST/GET/DELETE).
4. `require_user` dual-format.
5. UI Settings tab.
6. Verify: tạo token, gọi `/api/me/projects` bằng token, revoke, gọi lại 401.

## Open questions to research while building

- bcrypt vs argon2: bcrypt cố định work factor 10 OK, hay đầu tư argon2id ngay
  vì spec nó tốt hơn? Cost migrate sau là verify cũ + rehash.
- last_used update strategy: mỗi request immediate vs deferred 60s. Đo trước
  khi tối ưu.
- Có nên prefix khác cho admin token tương lai (`typa_...` for admin)? Phase 1
  không cần, ghi note.
