# RFC-006: Discord-native UI + shared projects

Status: **proposed** — implement after RFC-005.
Scope: tăng UX cho Activity, multi-user collab trong 1 Discord guild.

## Summary

App giờ là Discord Activity native. UI redesign theo constraint Activity SDK
(mobile-first, voice-aware optional). Project có owner + flag `shared`. Member
trong guild thấy project shared của nhau, ghim được làm bookmark cá nhân.
Phân quyền dựa trên Discord role (lấy lúc OAuth, lưu trong JWT).

## Why

1. **Multi-user**: hiện tại 1 admin tự quản. Cộng đồng VN ~1k user trong 1
   Discord server muốn cộng tác — owner chia sẻ project, member đọc.
2. **UI hiện tại sai persona**: sidebar 1 mục "Dự án" trống trải, header có
   nút rỗng, designed cho solo dev. Không match Discord Activity (mobile,
   sidebar mode, voice context).
3. **`users.tier` là duplicate Discord**: Discord đã có role system mạnh hơn.
   Source of truth nên là Discord, không phải DB column riêng.
4. **DMCA risk có thật** (B model: render xong member đọc trong app): tránh
   public discovery, không SEO, naming neutral.

## Non-goals (Phase 1)

- Bot service riêng, slash commands.
- Auto tạo Discord channel / category cho project.
- Channel post tự động, activity feed.
- Follow user, follow project social graph.
- Cross-guild discovery.
- Co-presence real-time (cursor sync, voice indicator).
- Voice channel co-watch.

## Schema

```sql
ALTER TABLE projects
  ADD COLUMN owner_id BIGINT REFERENCES users(id),
  ADD COLUMN shared   BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX idx_projects_owner_shared
  ON projects (owner_id) WHERE shared = TRUE;

CREATE TABLE project_pins (
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    pinned_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, project_id)
);
CREATE INDEX idx_project_pins_user ON project_pins(user_id);

-- Drop tier — Discord role là single source of truth.
ALTER TABLE users DROP COLUMN tier;
```

Bump `SCHEMA_VERSION` trong `postgres.py` từ `"1"` lên `"2"`.

## Permission

Discord role lấy lúc OAuth `code → JWT exchange`:

```python
# typoon/api/auth.py
async def fetch_member_roles(access_token, guild_id) -> list[str]:
    """GET /users/@me/guilds/{guild_id}/member → role IDs.
    Then resolve role IDs → role names via /guilds/{id}/roles (bot token)
    OR keep IDs and config maps role IDs.
    """
```

Strategy: lưu **role names** trong JWT (vì role name dễ đọc, ít đổi). JWT
claims:

```json
{
  "sub": "42",
  "iat": ...,
  "exp": ...,
  "roles": ["Admin", "Translator"]
}
```

Backend deps:

```python
def has_role(user, role_name): return role_name in user.get("roles", [])

async def require_admin(user = Depends(require_user), cfg = ...):
    if cfg.auth.admin_role_name not in user.get("roles", []):
        raise HTTPException(403)
    return user
```

Config thêm:

```toml
[auth]
admin_role_name = "Typoon Admin"
```

Trade-off: role thay đổi giữa session → user logout/login lấy lại. Chấp
nhận được.

Drop `users.tier`, drop `set_user_tier`, drop `bootstrap_discord_id` env
(thay bằng "tạo Discord role 'Typoon Admin' và gán cho user đầu tiên").

## API changes

```
# List filters
GET /api/projects                    → owned + shared (default, sorted by relevance)
GET /api/projects?filter=mine        → owner_id = me
GET /api/projects?filter=pinned      → joined với project_pins
GET /api/projects?filter=community   → shared=TRUE && owner_id != me

# Permissions
- read (list, detail, chapters, bubbles, pages, brief): owner OR shared
- mutation (upload, redo, edit, delete, settings, glossary): owner only

# Pin
POST   /api/projects/{id}/pin
DELETE /api/projects/{id}/pin

# Settings — share toggle (owner only)
PATCH /api/projects/{id}/settings  body: { shared?: bool, ... }
```

Project create gán `owner_id = current_user.id` tự động.

## UI changes

### Sidebar 3 mục

```
Brand (guild name + icon)
─────────────────────────
📁 Của tôi      (5)
⭐ Đã lưu       (3)
📖 Cộng đồng    (12)
─────────────────────────
⚙ Cài đặt
```

Routes:
- `/projects` — default = `?filter=mine` (mục "Của tôi" landing)
- `/projects?filter=pinned`
- `/projects?filter=community`

Filter active hiển thị highlight trong sidebar.

### Mobile-first redesign

Discord Activity chạy desktop + mobile + sidebar mode. Breakpoints:

- `<640px` (mobile + sidebar mode): sidebar collapse default, hamburger toggle.
- `>=640px` (desktop): sidebar visible.

Hiện tại Sidebar collapse logic OK, chỉ cần default-collapsed dưới 640px.

Header: hide search bar `<sm`, chỉ giữ user menu + workers indicator.

### Project card thêm:

- Star icon góc phải (toggle pin).
- Badge "Đã chia sẻ" nếu owner đang xem project shared của mình (chỉ trên trang chi tiết, không trên grid).

### Project detail page:

- Owner thấy toggle "Chia sẻ với cộng đồng" trong tab Settings.
- Member khác thấy: read-only view, không nút Upload / Edit / Delete / Glossary CRUD.

### Drop chrome rỗng

- Bell icon → ẩn (chưa có notification).
- ⌘K search palette → giữ nhưng wire vào `/api/search` thật, không placeholder.

## Order of work

Mỗi step compile + type-check + visual smoke OK trước khi qua step kế.

1. **Schema + storage layer**: thêm `owner_id`, `shared`, `project_pins`, drop `tier`. Bump SCHEMA_VERSION → 2.
2. **Storage methods**: `list_projects(owner_id=, filter=, viewer_id=)`, `pin_project`, `unpin_project`, `is_pinned`.
3. **Auth**: fetch roles trong OAuth exchange, lưu vào JWT. Drop `tier` & `bootstrap_discord_id`. Add `admin_role_name` config.
4. **API routes**: list filter, pin endpoints, share trong settings, permission check (owner-only mutations, owner-or-shared reads).
5. **Web API client**: thêm filter param, pin endpoints, owner_id field.
6. **Web UI**: sidebar 3 mục, route filter, pin star, share toggle, hide chrome rỗng.
7. **Mobile breakpoint**: sidebar default-collapsed `<640px`, header gọn.
8. **Verify visual**: chạy tay, screenshot 3 mục sidebar, mobile view, share toggle.

## Risks

- **Role lookup delay**: OAuth exchange gọi thêm Discord API → +200-500ms login. Acceptable.
- **JWT size grow**: thêm `roles` array. ~20 bytes/role. Phase 1 max 5 role → OK.
- **Migration `users.tier` drop**: Phase 1 có data thật (admin). Drop column = mất info. Cần đảm bảo `admin_role_name` đã set trong Discord trước khi flip schema.
- **Project owner null cho project cũ**: project tạo trước RFC-006 chưa có owner. Migration: gán `owner_id = NULL` cho các project cũ, treat NULL như "system-owned" — không ai mutate được trừ Discord admin role. Hoặc backfill từ `created_at` (admin nào tạo lúc đó).

## Out of scope explicitly

- Discord bot, slash commands.
- Auto channel/category.
- Comment/discussion (dùng Discord channel của owner).
- Notification system.
- Activity feed.
- Credit/billing (RFC-007).
- Mobile co-presence.
