# RFC-009: Browser extension — MangaDex import

Status: **proposed** — implement after RFC-008 (depends on API tokens).
Scope: extension Chrome/Firefox cho user import chapter từ MangaDex vào
Typoon project mà không rời browser.

## Summary

Extension popup nhỏ. User paste API token 1 lần. Khi đang ở page MangaDex
(title hoặc chapter), popup hiện project dropdown + form import. Service
worker fetch pages từ MangaDex API public, pack một zip, multipart-PUT
lên inbox storage qua presigned URLs (`/chapters/upload-init` +
`/chapters/upload-finalize`). Engine sau đó fetch zip từ inbox và ingest;
luồng này không còn đi qua CF Tunnel → không bị chiếm băng thông
upstream của engine.

## Why

- Audience Phase 1 không terminal-friendly → CLI không phải first tool.
- Workflow tự nhiên: user đang đọc MangaDex, thấy chap hay, 1 click vào
  Typoon dịch.
- MangaDex có API public ổn định, không cần scrape DOM, không CF challenge,
  ToS cho phép.

## Non-goals (Phase 1)

- Site khác MangaDex (Webtoons, Lezhin, NetTruyen, …). Mỗi site = 1 adapter,
  thêm sau.
- Auto detect chapter cập nhật mới và push (passive monitoring).
- Two-way sync (chỉnh translation từ extension).
- Login OAuth qua extension. Token paste là đủ.
- Submit lên Chrome Web Store / Firefox Addons. Self-host ZIP, user
  "Load unpacked".
- Mobile (Android Chrome có support extension nhưng audience không đáng).

## Stack

Quyết định cứng:
- **Manifest V3** (Chrome ≥88, Firefox ≥109).
- **TypeScript**.
- **Vite + `@crxjs/vite-plugin`** — research khi build (alternative: webpack
  + crx-hotreload). crxjs là default, đổi nếu thấy maintainence active hơn.
- **React + Tailwind** trong popup, reuse design tokens từ web SPA.
- **chrome.storage.local** cho config (`api_url`, `token`,
  `last_project_id`).

## Repo location

`ext/` ở root repo này.

```
typoon/v2/
  typoon/        — engine
  web/           — SPA
  ext/           — extension (mới)
    manifest.json
    src/
      popup/
      background/
      content/
      lib/
    package.json
    vite.config.ts
```

Lý do cùng repo:
- Share docs, version sync với engine.
- Khi sửa API contract (engine), sửa cả ext trong cùng PR.
- Dev cycle ngắn.

Khi ext có release cycle độc lập (Phase 2+, submit store), tách repo riêng.

## File structure (đề xuất, finalize khi build)

```
ext/
  manifest.json
  src/
    popup/
      Popup.tsx              ← UI hiện khi click icon
      ProjectPicker.tsx
      ChapterForm.tsx
      MangaForm.tsx          ← title page mode
      Setup.tsx              ← lần đầu paste token
    background/
      worker.ts              ← service worker
      queue.ts               ← upload queue, retry, progress events
    content/
      mangadex.ts            ← detect title/chapter trên page MangaDex
                              (chỉ inject vào *.mangadex.org)
    lib/
      api/
        typoon.ts            ← engine client (POST upload, GET projects)
        mangadex.ts          ← MangaDex API client
      storage.ts             ← chrome.storage wrappers
      ratelimit.ts           ← token bucket cho MangaDex API
  vite.config.ts
  package.json
  tsconfig.json
```

## Engine API used

Phase 1 cần engine expose:

| Endpoint | Status | Note |
|---|---|---|
| `POST /api/me/tokens` | RFC-008 mới | UI tạo token để paste vào ext |
| `GET /api/me/projects` | mới (alias `/api/projects?filter=mine`) | dropdown project |
| `POST /api/projects` | có | "Tạo dự án mới" trong popup |
| `POST /api/projects/{id}/chapters/upload-init`     | mới | presign multipart parts |
| `POST /api/projects/{id}/chapters/upload-finalize` | mới | complete + ingest |
| `POST /api/projects/{id}/chapters/upload-abort`    | mới | huy upload đang dở |
| `GET /api/projects/{id}/chapters` | có | hide chapter idx đã có để tránh duplicate |

Không cần thêm endpoint nào khác Phase 1.

## UX flows

### Flow A: lần đầu setup

User cài extension → click icon:

```
┌─ Typoon ──────────────────┐
│ Kết nối Typoon            │
│                           │
│ API URL                   │
│ [https://typoon.tld     ] │
│                           │
│ API Token                 │
│ [typ_____________________]│
│ Lấy token ở Typoon →      │
│   Cài đặt → API tokens    │
│                           │
│ [Lưu]                     │
└───────────────────────────┘
```

Click [Lưu]:
1. GET `{api_url}/api/me/projects` với token → verify auth.
2. Lưu config vào `chrome.storage.local`.
3. Switch sang main UI.

### Flow B: page MangaDex chapter

URL pattern: `https://mangadex.org/chapter/{uuid}`. Content script detect,
gửi page metadata về service worker:

```
┌─ Typoon ──────────────────────┐
│ Solo Leveling — Ch.42          │
│ phép thuật cấm                 │
│                                │
│ Project                        │
│ [Solo Leveling          ▾]     │ ← /api/me/projects, dropdown
│ [+ Tạo project mới]            │
│                                │
│ Số chương: [42_____]           │ ← prefill
│ Tên:       [phép thuật cấm__] │ ← prefill từ chapter title
│                                │
│ [⬇ Tải + Upload]               │
└────────────────────────────────┘
```

Click [Tải + Upload]:
1. Fetch chapter pages từ MangaDex at-home server.
2. Pack `FormData` với mỗi page là 1 file image.
3. POST `/api/projects/{id}/chapters/upload-init` → PUT từng part zip
   lên presigned URL → POST `/api/projects/{id}/chapters/upload-finalize`.
   SDK chia sẻ ở `packages/upload-sdk` loàn driver multipart + progress
   + retry; web SPA và ext dùng chung.
4. Hiện progress trong popup; nếu user đóng popup, service worker tiếp tục
   chạy + notify khi xong.

### Flow C: page MangaDex title (multi)

URL pattern: `https://mangadex.org/title/{uuid}`. Content script extract
manga ID + title:

```
┌─ Typoon ──────────────────────┐
│ Solo Leveling                  │
│ Chu Gong • 179 chương          │
│                                │
│ Project                        │
│ [Solo Leveling          ▾]     │
│                                │
│ Ngôn ngữ                       │
│ [Tiếng Anh              ▾]     │ ← từ /manga/{id}/feed available langs
│                                │
│ Nhóm dịch                      │
│ [Tất cả                 ▾]     │ ← từ feed groups
│                                │
│ Phạm vi                        │
│ Từ [ 1] đến [10]               │
│ ☐ Bỏ qua chapter đã có         │
│                                │
│ [⬇ Tải + Upload (10 chương)]   │
└────────────────────────────────┘
```

Multi flow:
1. List chapters từ `/manga/{id}/feed?translatedLanguage[]=...`.
2. Filter theo group + range.
3. Fetch + upload tuần tự (rate limit 5 req/s MangaDex).
4. Progress UI hiện row per chapter:
   ```
   ch.1   ✓ uploaded
   ch.2   ⏳ downloading 12/18
   ch.3   · queued
   ```

### Flow D: page khác (không phải MangaDex)

Popup hiện:
```
┌─ Typoon ──────────────────────┐
│ Trang này chưa được hỗ trợ.    │
│                                │
│ Phase 1 chỉ hỗ trợ MangaDex.   │
│ Bạn vẫn có thể upload thủ công │
│ qua web Typoon.                │
│                                │
│ [Mở Typoon]                    │
└────────────────────────────────┘
```

## Permissions

```json
{
  "manifest_version": 3,
  "name": "Typoon Importer",
  "version": "0.1.0",
  "permissions": ["storage"],
  "host_permissions": [
    "https://api.mangadex.org/*",
    "https://*.mangadex.network/*",
    "https://uploads.mangadex.org/*"
  ],
  "optional_host_permissions": [
    "https://*/*"
  ],
  "action": { "default_popup": "popup.html" },
  "background": { "service_worker": "background.js" },
  "content_scripts": [{
    "matches": ["https://mangadex.org/*"],
    "js": ["content/mangadex.js"]
  }]
}
```

`optional_host_permissions` cho engine URL — user nhập URL ở Setup, ext
prompt browser xin permission cho host đó. Cách này cho phép self-host
engine nhiều domain khác nhau mà không cần manifest cứng.

## MangaDex API research notes

Endpoint cần (research khi build, version có thể đổi):

- `GET /manga/{uuid}` — manga info (title, lang, …)
- `GET /manga/{uuid}/feed?translatedLanguage[]=&order[chapter]=asc&limit=100`
  — paginate chapter list, max 500 limit/req.
- `GET /manga/{uuid}/aggregate` — chapter tree by volume → quick group/lang
  count.
- `GET /at-home/server/{chapter_id}` — get base URL + filename array (15min
  TTL).
- Image: `{base_url}/data/{hash}/{filename}` (full quality) hoặc
  `data-saver/...` (compressed).

Rate limit: docs nói "no auth needed for read endpoints, common sense throttle".
Token bucket 5 req/s implement trong `lib/ratelimit.ts`.

Version: pin `https://api.mangadex.org/v5/...`? Research khi build — docs
hiện hành dùng base URL không version, nhưng có thể có version gateway. Note
trong code ngày test.

## Upload to engine

Tài nguyên upload nằm ở `packages/upload-sdk` (workspace bun, share với web
SPA). SW import `uploadChapterZip` + `packPagesToZip`, fetch tất cả phân
tử CDN song song, pack store-mode zip, upload multipart lên inbox
(S3-compat: R2/S3/B2/MinIO/Wasabi). Engine sau đó fetch zip từ inbox và
chuẩn bị chapter; băng thông upstream của engine không bị ăn.

```ts
import { uploadChapterZip, packPagesToZip } from '@typoon/upload-sdk'
import { TypoonClient } from '@core/typoon'

async function uploadChapter(opts: {
  apiUrl: string,
  token:  string,
  projectId: number,
  pages:  Uint8Array[],   // raw bytes đã fetch từ CDN
  number?: string,
  title?:  string,
  onProgress?: (p: { bytesSent: number; bytesTotal: number; speedBps?: number }) => void,
}) {
  const client = new TypoonClient({ apiUrl: opts.apiUrl, token: opts.token })
  const zip = packPagesToZip(opts.pages.map((bytes, i) => ({
    source: `${i.toString().padStart(4, '0')}.jpg`,
    bytes,
  })))
  return uploadChapterZip(client, opts.projectId, zip, {
    number: opts.number,
    title:  opts.title,
    onProgress: opts.onProgress,
  })
}
```

Engine multipart hỗ trợ nhiều image files → image set mode (đã có).

## Progress + state

Service worker giữ queue + progress trong memory + sync xuống
`chrome.storage.session`. Popup mở lại đọc state hiện hành.

Background nếu MV3 service worker idle bị kill thì lifecycle: user click
[Tải] → SW spin up → upload → queue empty → SW có thể bị kill. Đóng popup
giữa chừng OK vì SW vẫn chạy đến khi network idle.

Alternative chắc chắn hơn: dùng `chrome.alarms` keep-alive 25s ping. Research
khi build, đo SW lifecycle trên Chrome stable.

## Notification khi xong

- Tab popup đang mở → update UI inline.
- Tab popup đóng → `chrome.notifications.create()` system notification:
  "Solo Leveling Ch.42 đã upload".

## Distribute Phase 1

1. Build: `npm run build` → `ext/dist/` chứa unpacked extension.
2. ZIP: `cd ext && zip -r typoon-ext-v0.1.0.zip dist/`.
3. Upload ZIP lên Discord channel guild (admin-only post hoặc pin).
4. Hướng dẫn user trong README ngắn:
   ```
   1. Download typoon-ext.zip, giải nén ra folder.
   2. Mở Chrome → chrome://extensions → bật Developer mode.
   3. Load unpacked → chọn folder vừa giải nén.
   4. Click icon Typoon trên thanh extension → paste API token.
   ```

Auto-update: Phase 1 không. User upgrade thủ công (thông báo trong Discord
khi có version mới).

Phase 2: submit Chrome Web Store (review delay 1-2 tuần, có content policy
risk vì target nội dung manga). Đánh giá lại lúc đó.

## Order of work

1. Scaffold `ext/` với Vite + crxjs + React.
2. Setup screen + chrome.storage wiring.
3. Engine API client trong ext (auth verify, list projects).
4. MangaDex API client (chapter info, at-home server, fetch pages).
5. Content script detect MangaDex page → message tới SW.
6. Popup chapter mode (single chapter upload).
7. Popup title mode (range upload với queue + progress).
8. System notification khi upload xong.
9. Manual test: load unpacked, login, import 1 chapter, import 10 chapter.
10. README cho user install.

Mỗi step compile, test thủ công trên Chrome stable. Không có unit test —
extension UI test setup overhead vượt giá trị Phase 1.

## Risks

- **MV3 service worker lifecycle**: SW có thể bị kill giữa upload. Mitigate:
  keep-alive alarm + retry queue persist `chrome.storage.session`.
- **MangaDex API thay đổi**: schema endpoint đổi, ext break. Mitigate: version
  ext trong manifest, prompt user update khi engine detect ext version cũ
  (Phase 2).
- **CORS engine**: SPA dev origin `http://localhost:5173` đã trong allowlist.
  Extension origin là `chrome-extension://<id>` — phải thêm vào CORS
  allowlist hoặc dùng `allow_origin_regex`. Test khi build, fix engine config
  nếu cần.
- **Token leak qua DevTools**: user mở DevTools popup thấy token trong
  `chrome.storage`. Acceptable risk — token có thể revoke. Document trong
  README.
- **Manga ID detection**: URL pattern MangaDex có thể thay đổi (slug vs uuid).
  Test khi build.
- **Image format**: MangaDex serve jpeg/png/webp. Engine `/upload` accept
  multi-image, sources/upload phải handle các format này (đã có vì PIL).
- **DMCA**: extension fetch nội dung MangaDex từ máy user, engine nhận
  multipart không biết nguồn. Risk pháp lý không tăng so với manual upload
  ZIP.

## Open research khi build

- crxjs vs alternative bundler (Plasmo, WXT). crxjs is most popular cho Vite,
  test trước.
- Service worker keep-alive pattern: alarm vs offscreen document API.
- MangaDex API rate limit thực tế (5 req/s là lý thuyết, khi nào bị 429).
- At-home server token expire — refresh strategy.
- Chrome extension Manifest V3 trên Firefox khác nhau gì (event page → SW
  difference).
- Notification permission UX (auto-prompt vs user-action-triggered).
- Image deduplication: nếu user import lại chapter đã có, engine có nên
  reject 409 hay ghi đè? Hiện engine `get_or_create_chapter` re-use, ghi
  đè artifact. UI ext nên warn "Chương đã có, ghi đè?".

## What ships when

Phase 1.0 (RFC-008 + RFC-009):
- API tokens infra.
- Extension: setup, MangaDex single chapter import, basic multi.

Phase 1.1 (sau beta):
- Auto-detect chapter mới chưa import.
- Better progress UI (system tray, persistent queue).
- Retry logic robust hơn.

Phase 2 (xa):
- Adapter cho site khác.
- Web Store submission.
- 2-way sync (xem translation status từ ext).
