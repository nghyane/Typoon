# Typoon Refactor Plan — COMPLETED

> 本文档用于新 session 的上下文传递。所有指令以本文为准，不要假设未记录的内容。

---

## 一、项目概况

### 技术栈
- **前端**: React + TanStack Router + TanStack Query + Vite
- **后端**: Cloudflare Workers (Hono) + D1 (SQLite) + R2 (对象存储)
- **WebSocket**: Durable Object (TranslationStatusDO) 实时推送翻译进度

### 目录结构
```
web/src/
├── shared/api/api.ts      ← API 客户端（已重构）
├── store/                 ← React Query store 层
├── features/              ← UI 组件（不改动设计系统）
├── routes/                ← TanStack Router 页面
├── app/                   ← Layout、providers、设计系统（不改动）
└── index.css              ← Tailwind 设计 token（不改动）

workers/api/src/
├── index.ts               ← 已部署的 API 入口
├── routes/                ← 路由定义（已重构 + 简化）
├── store/                 ← D1 查询层
├── middleware/            ← auth、error、pagination
├── types.ts               ← API 响应类型定义
└── do/ + rpc/             ← Durable Object + RPC 回调
```

---

## 二、已完成工作

### Backend 变更

#### 已简化 `workers/api/src/routes/translations.ts`
- **Cut**: `GET /:id/bubbles`, `PUT /:id/edits`, `DELETE /:id/edits/:page/:bubble`
- **Pattern**: Sub-router `/:id` với middleware `requireTranslationAccess`
- **Middleware**: Parse id + fetch detail + verify access → `ctx.set("translationDetail")`
- **Keep**: `POST /`, `GET /`, `GET /:id`, `POST /:id/redo`, `DELETE /:id`, `WS /:id/ws`

#### 已简化 `workers/api/src/store/translations.ts`
- **Cut**: `upsertTranslationEdit`, `getTranslationEdits`, `deleteTranslationEdit`, `TranslationEditRow`
- **Cut**: `has_edits` field từ `TranslationDetail`

#### 已简化 `workers/api/src/types.ts`
- **Cut**: `ApiBubbleEdit` interface
- **Cut**: `has_edits` field từ `ApiTranslation`
- **Add**: `translationDetail` optional field to `ContextVars`

### Frontend 变更

#### `web/vite.config.ts`
- **Added**: `ws: true` to `/api` proxy for WebSocket support

#### `web/src/shared/api/api.ts`
- **Path updates**: `/material/*` → `/materials/*`, `/translate/*` → `/translations/*`, `/community/recent` → `/community/feed`, `/admin/ops/*` → `/admin/*`
- **Cut methods**: `createLocalMaterial`, `enrichMaterialMetadata`, `listTranslationBubbles`, `patchTranslation`, `upsertEdit`, `deleteEdit`, `workers`
- **Cut admin methods**: `requeueTask`, `releaseTask`, `forceFailTask`, `listActions`
- **New admin methods**: `retryTask(taskId, body)`, `deleteTask(taskId)`
- **Type updates**: `listCommunityFeed` returns `{ items, next_cursor }`, removed `ApiBubbleEdit`, removed `has_edits` from `ApiTranslation`

#### `web/src/routes/index.tsx`
- `api.listCommunityRecent` → `api.listCommunityFeed`
- `community.data` → `community.data?.items`

#### `web/src/routes/admin.ops.tsx`
- `api.workers` → `api.adminOps.listTasks` for queue stats
- `requeueTask/releaseTask/forceFailTask` → `retryTask/deleteTask`
- Removed AuditPanel (backend has no actions endpoint)
- Removed unused imports: `ApiAdminAction`, `AdminActionKind`, `ACTION_LABEL`, `Unlock`

#### `web/src/app/WorkersIndicator.tsx`
- `api.workers` → `api.adminOps.listTasks`
- Simplified to aggregate from task list instead of queue stats

#### `web/src/features/link/useAutoEnrichWork.ts`
- `api.enrichMaterialMetadata` → `api.patchMaterial` (strips `source_signals`)

---

## 三、API Route Mapping (Final)

| Old Path | New Path | Status |
|---|---|---|
| `POST /material/import` | `POST /materials/import` | ✅ |
| `POST /material` | removed | ✅ |
| `PATCH /material/:id` | `PATCH /materials/:id` | ✅ |
| `DELETE /material/:id` | `DELETE /materials/:id` | ✅ |
| `POST /material/:id/cover` | `POST /materials/:id/cover` | ✅ |
| `POST /material/:id/enrich-metadata` | merged into `PATCH /materials/:id` | ✅ |
| `POST /translate` | `POST /translations` | ✅ |
| `GET /translate/mine` | `GET /translations` | ✅ |
| `GET /translate/:id` | `GET /translations/:id` | ✅ |
| `GET /translate/:id/bubbles` | **removed** | ✅ |
| `PUT /translate/:id/edits` | **removed** | ✅ |
| `DELETE /translate/:id/edits/:p/:b` | **removed** | ✅ |
| `POST /translate/:id/redo` | `POST /translations/:id/redo` | ✅ |
| `DELETE /translate/:id` | `DELETE /translations/:id` | ✅ |
| `GET /community/recent` | `GET /community/feed` | ✅ |
| `GET /workers` | **removed** | ✅ |
| `/admin/ops/stages` | `/admin/stages` | ✅ |
| `/admin/ops/tasks` | `/admin/tasks` | ✅ |
| `/admin/ops/tasks/.../requeue` | `POST /admin/tasks/:id/retry` | ✅ |
| `/admin/ops/tasks/.../release` | **removed** | ✅ |
| `/admin/ops/tasks/.../fail` | `DELETE /admin/tasks/:id` | ✅ |
| `/admin/ops/drafts/:id/restart` | `/admin/drafts/:id/restart` | ✅ |
| `/admin/ops/actions` | **removed** (pending backend) | ✅ |

---

## 四、Verification

```bash
# Backend TypeScript
cd workers/api && ./node_modules/.bin/tsc --noEmit  # ✅ EXIT: 0

# Frontend TypeScript
cd web && ./node_modules/.bin/tsc --noEmit          # ✅ EXIT: 0

# No old paths remaining
grep -rn "/material/import\|/translate/\|/community/recent\|/workers\|/admin/ops/" web/src/shared/api/api.ts
# ✅ 0 matches (excluding upload/memory routes which are separate)

# No removed method references
grep -rn "api\.workers\|api\.enrichMaterialMetadata\|api\.listTranslationBubbles\|api\.upsertEdit" web/src/
# ✅ 0 matches
```

---

## 五、Pending (not in scope)

- Upload routes (`/material/:id/chapter/upload-*`) — backend uses `/uploads/presign` + `/uploads/finalize` instead, but frontend upload flow is separate
- Admin audit actions endpoint — backend needs `GET /admin/actions` to restore AuditPanel
- `POST /materials/import` requires `work_id` in body — frontend `buildMaterialPayload` needs update (caller must provide work_id)
