# Reverse-engineering manga sources

Playbook để viết 1 source manifest mới (`packages/manga-sources/{id}.json`) trong 30–60 phút. Đúc kết từ HappyMH; áp dụng nguyên cho mọi site SSR + SPA tương tự.

---

## 0. Mục tiêu cần lấy

Mỗi site cần 5 endpoint (xem `packages/manga-sources/README.md` schema):

| Endpoint     | Trả về                                         | Thường ở dạng    |
|--------------|------------------------------------------------|------------------|
| `popular`    | List manga (cover, title, URL)                 | HTML SSR         |
| `latest`     | List manga                                     | HTML SSR         |
| `manga`      | Detail (title, cover, desc, …)                 | HTML SSR         |
| `chaptersApi`| Danh sách chapter (id, name, order, …)         | JSON AJAX        |
| `chapter`    | Danh sách trang ảnh                            | JSON AJAX        |

Cả 5 đều đi qua **`/cdn/c/{host}{path}`** (bunle-cdn) — đã inject Referer/UA per-host, đã pass-through Range, status, content-type. Không cần cookie trừ khi site bắt buộc.

---

## 1. Setup môi trường reverse

**Edge + CDP**. Browser thật để có sẵn cookie/JS context — phương án nhanh nhất.

```bash
# Kill any existing instance using port 9222
EDGE="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
PROFILE=/tmp/typoon-edge-profile
mkdir -p "$PROFILE"
"$EDGE" \
  --remote-debugging-port=9222 \
  --remote-allow-origins='*' \
  --user-data-dir="$PROFILE" \
  --no-first-run --no-default-browser-check \
  "https://<site>/manga/<sample-slug>" &
```

**Tại sao `--remote-allow-origins='*'`**: thiếu nó → mọi WS connect bị 403. Phải set, không có cách khác.

**Tại sao profile riêng (`/tmp/typoon-edge-profile`)**: profile cá nhân của Edge (instance khác) chiếm port 9222 trước, lệnh `kill` không đụng nó. Profile riêng = process riêng, dễ debug.

Pass Cloudflare challenge thủ công 1 lần — cookie sống trong profile, các script CDP sau đó dùng được.

---

## 2. Helper CDP — `/tmp/cdp.py`

Raw WebSocket, không phụ thuộc `agent-browser` (vốn tự tạo tab và bám sai tab).

```python
"""CDP eval — raw socket. Usage: python3 cdp.py <ws_url> <expression>"""
import json, socket, base64, os, struct, sys
ws_url = sys.argv[1]; expr = sys.argv[2]
host_path = ws_url[5:]; host, path = host_path.split("/", 1)
hostname, port = host.split(":"); path = "/" + path
sock = socket.create_connection((hostname, int(port)), timeout=15)
key = base64.b64encode(os.urandom(16)).decode()
sock.send(f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
          f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
          f"Sec-WebSocket-Version: 13\r\n\r\n".encode())
buf = b""
while b"\r\n\r\n" not in buf: buf += sock.recv(4096)

def send(payload):
    h = bytes([0x81]); p = len(payload); mask = os.urandom(4)
    if p < 126:        h += bytes([0x80|p])
    elif p < 65536:    h += bytes([0x80|126]) + struct.pack(">H", p)
    else:              h += bytes([0x80|127]) + struct.pack(">Q", p)
    h += mask
    sock.send(h + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

def recv():
    h = b""
    while len(h) < 2: h += sock.recv(2 - len(h))
    plen = h[1] & 0x7F
    if plen == 126: plen = struct.unpack(">H", sock.recv(2))[0]
    elif plen == 127: plen = struct.unpack(">Q", sock.recv(8))[0]
    d = b""
    while len(d) < plen: d += sock.recv(plen - len(d))
    return json.loads(d.decode())

send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                 "params": {"expression": expr, "awaitPromise": True,
                            "returnByValue": True}}).encode())
r = recv().get("result", {}).get("result", {})
print(r.get("value") if "value" in r
      else "ERR: " + r.get("description", json.dumps(r)))
```

Và `/tmp/cdp_capture.py` để capture network requests (dùng khi cần xem **tham số ẩn** mà SPA tự thêm):

```python
"""Open tab, enable Network domain, navigate, dump matching requests."""
import json, socket, base64, os, struct, sys, time
# … same WS handshake as above …
def cmd(method, params=None):
    cmd.id = getattr(cmd, "id", 0) + 1
    send(json.dumps({"id": cmd.id, "method": method,
                     "params": params or {}}).encode())
cmd("Network.enable"); cmd("Page.enable")
cmd("Page.navigate", {"url": sys.argv[2]})
end = time.time() + int(sys.argv[3] if len(sys.argv) > 3 else 8)
reqs = {}
while time.time() < end:
    r = recv_with_timeout(1)
    if not r: continue
    if r.get("method") == "Network.requestWillBeSent":
        u = r["params"]["request"]["url"]
        if any(s in u for s in ("/apis/", "/manga/", "/v2.")):
            reqs[r["params"]["requestId"]] = u
    elif r.get("method") == "Network.responseReceived":
        rid = r["params"]["requestId"]
        if rid in reqs:
            reqs[rid] = f"{r['params']['response']['status']} {reqs[rid]}"
for v in reqs.values(): print(v)
```

Lấy tab id từ `curl http://127.0.0.1:9222/json | jq` (filter `type == page`).

```bash
TAB=$(curl -s http://127.0.0.1:9222/json | python3 -c "
import sys,json
[print(t['id']) for t in json.load(sys.stdin)
 if t.get('type')=='page' and '<site>' in t.get('url','')][:1]")
WS="ws://127.0.0.1:9222/devtools/page/$TAB"
```

---

## 3. Quy trình recon — order chuẩn

### 3.1 Verify proxy

```bash
# rank/popular landing
curl -sI "https://927251094806098001.discordsays.com/cdn/c/<site>/" | head -5
```

Trả 200 → proxy OK + Referer/UA inject pass hotlink. Trả 403/404 → site cần host-profile mới trong `bunle-cdn/functions/[[path]].ts` (`PROFILES`).

### 3.2 SSR pages — selector dump

```bash
curl -s "https://927251094806098001.discordsays.com/cdn/c/<site>/<list-path>" -o /tmp/page.html
# Class names có chữ list/item/manga/book/comic/rank
grep -oE 'class="[^"]+"' /tmp/page.html | sort -u | grep -iE "list|item|manga|book|comic|rank|cover|title" | head -20
# Spot-check 1 row
python3 -c "
import re
html = open('/tmp/page.html').read()
m = re.search(r'<[^>]+class=\"<row-class>\"[^>]*>(.*?)</div>', html, re.DOTALL)
print(m.group(0)[:1500])
"
```

→ chốt `list` selector + `fields { url, title, cover }`.

### 3.3 SPA pages — capture network

Trang SPA có **HTML stub + JS render**. Đừng cố scrape HTML. Capture request thực:

```bash
python3 /tmp/cdp_capture.py "$WS" "https://<site>/mangaread/<code>/<cid>" 10
```

Output sẽ liệt kê mọi XHR. Tìm endpoint trả JSON list trang ảnh — thường là `/v2.0/apis/manga/...` hoặc `/api/chapter/...`.

### 3.4 Nếu API trả `400 expired` hoặc lỗi lạ

Đừng đoán cookie/sign — capture request thực, **so sánh** với fetch của bạn:

```python
# Trong tab debug:
python3 /tmp/cdp.py "$WS" "
fetch('<api-url>?<params>', {credentials:'include',
  headers:{'X-Requested-With':'XMLHttpRequest'}})
  .then(r=>r.text()).then(t=>t.slice(0,500))
"
```

Khác biệt = magic param/header bị thiếu. Pattern hay gặp:

- **`?v=v4.203411`** — site-wide version constant, hardcode trong chunk JS (HappyMH).
- **`?_t=<ms>`** — cache buster, luôn cần.
- **`X-Requested-With: XMLHttpRequest`** — anti-bot, luôn cần.
- **`Referer`** — proxy đã tự inject theo host profile, hiếm khi cần override.

### 3.5 Tìm magic constants trong JS

```bash
# Dump tất cả chunk.js đã load
python3 /tmp/cdp.py "$WS" "JSON.stringify(performance.getEntries().map(e=>e.name).filter(n=>n.includes('.js')))" \
  | tr ',' '\n' | grep -oE 'https?://[^"]+\.js' | sort -u > /tmp/jslist.txt

# Grep từng cái
while read url; do
  curl -s "https://927251094806098001.discordsays.com/cdn/c/${url#https://}" -o /tmp/c.js
  if grep -q "<MAGIC>" /tmp/c.js; then
    echo "FOUND in ${url##*/}"
    grep -oE ".{50}<MAGIC>.{50}" /tmp/c.js | head -2
  fi
done < /tmp/jslist.txt
```

Nếu mã obfuscated (string array + decoder):

```js
// Cut chunk lên đến điểm rotation IIFE chạy xong, return decoder N:
const cut = src.indexOf(`F(),((()=>{`);  // pattern of webpack-obfuscator
const N = new Function(src.slice(0, cut) + '; return N;')();
console.log(N(0xb6c));  // -> "/v2.0/apis/manga/reading"
```

Mỗi module rewires decoder với cùng base offset → 1 lần extract N là decode được mọi `fK(0xN)` toàn file.

---

## 4. Viết manifest — checklist

```jsonc
{
  "id":       "<id>",              // kebab-case, ổn định
  "host":     "<m.example.com>",   // primary host, ăn vào allowlist proxy
  "language": "zh",
  "version":  "0.1",
  "endpoints": {
    "popular":     { /* SSR HTML list */ },
    "latest":      { /* SSR HTML list */ },
    "manga":       {
      "url":     "{mangaUrl}",
      "parse":   "html",
      "extract": "/manga/(?<code>[^/?#]+)",   // pull vars from URL for chaptersApi
      "fields":  { "title": ".x", "cover": "img@src", ... }
    },
    "chaptersApi": {
      "url":     "<json-api>?code={code}&...",
      "parse":   "json",
      "headers": { "X-Requested-With": "XMLHttpRequest" },
      "list":    "$.data.items",
      "fields":  {
        "id":     "@id",
        "number": "@order",
        "title":  "@chapterName",
        "url":    "=https://<site>/mangaread/{code}/{id}"   // template
      }
    },
    "chapter": {
      "url":     "<json-api>?code={code}&cid={cid}&v=<magic>",
      "extract": "/mangaread/(?<code>[^/?#]+)/(?<cid>[^/?#]+)",
      "parse":   "json",
      "headers": { "X-Requested-With": "XMLHttpRequest" },
      "list":    "$.data.scans",
      "fields":  { "url": "@url" }
    }
  }
}
```

**Selector grammar** (`web/src/features/browse/manifest/selectors.ts`):
- `selector` → text của match đầu
- `selector@attr` → attribute
- `@attr` → attribute của root hiện tại (row context)
- `$.json.path[*]` → JSONPath (subset: `$`, `.key`, `[N]`, `[*]`)
- `=template-with-{vars}` → composite, dùng 2-pass resolve (selectors trước, templates sau)

**Template vars có sẵn**:
- `{mangaUrl}`, `{chapterUrl}` — input gốc
- `{<group>}` từ `extract` regex
- `{<key>}` từ các field selector đã resolved trong cùng row

---

## 5. Test manifest

```bash
# 1. Hot reload Vite — không cần build lại
# 2. Mở /browse/{id} → 100 manga
# 3. Click 1 manga → chapters load qua chaptersApi
# 4. Click 1 chapter → reader load qua chapter endpoint
```

Nếu list rỗng → kiểm tra:
- Selector sai (mở DevTools, `document.querySelectorAll('<list>').length`)
- JSON path sai (`fetch(url).then(r=>r.json()).then(d => d.<path>)`)
- CORS preflight rớt header → `bunle-cdn` đã set `Access-Control-Allow-Headers: *`, không cần list từng cái

---

## 6. Sai lầm cần TRÁNH

Đây là những thứ đã thử mà **mất thời gian, không cần thiết**:

| Sai lầm                                                    | Đúng                                                       |
|------------------------------------------------------------|------------------------------------------------------------|
| Reverse cookie session (yH/yb GA-cookie algo)              | Hầu hết "page expired" là thiếu **query param** không cookie. Capture network trước. |
| Dùng `agent-browser open` để test                          | Nó tự tạo tab mới, không bám tab bạn xem. Dùng `cdp.py` raw WS với tab id từ `/json`. |
| Trông cậy `playwright connect_over_cdp`                    | Tab list chỉ thấy contexts mới tạo. Raw `/json/list` thấy toàn bộ.|
| Curl direct upstream (`m.happymh.com/...`) để verify       | Cloudflare block 403. Luôn đi qua `/cdn/c/...` proxy. |
| Tin `agent-browser tab list` là source of truth            | Nó chỉ list tab playwright attach. Dùng `curl /json` trực tiếp. |
| Đoán "session cookie" khi thấy "expired"                   | Captured request sẽ lộ tham số thực (đối với HappyMH là `v=v4.X`). |
| Allow CORS từ Edge bằng `--remote-allow-origins=` (thiếu `*`)| Edge yêu cầu `*` hoặc match exact origin. Cứ dùng `*`. |
| `Access-Control-Allow-Headers: X-Proxy-*` (suffix wildcard)| Browser không hỗ trợ suffix. Dùng `*` (an toàn vì proxy không nhận credentials). |
| `import.meta.env.VITE_PUBLIC_BASE_URL` thay đổi giữa dev/prod | Pattern thống nhất: trong DA → same-origin, ngoài → env var. Xem `api.ts` + `browse/proxy.ts`. |

---

## 7. Reference: HappyMH endpoints (đã verify)

| Mục đích       | Method | URL                                                                              | Headers                              |
|----------------|--------|----------------------------------------------------------------------------------|--------------------------------------|
| Popular        | GET    | `/rank/day` (HTML)                                                               | UA default                           |
| Latest         | GET    | `/rank/week` (HTML)                                                              | UA default                           |
| Manga          | GET    | `/manga/{code}` (HTML)                                                           | UA default                           |
| Chapter list   | GET    | `/v2.0/apis/manga/chapterByPage?code={code}&page=1&order=desc&_t={ms}`           | `X-Requested-With: XMLHttpRequest`   |
| Chapter pages  | GET    | `/v2.0/apis/manga/reading?code={code}&cid={cid}&v=v4.203411&_t={ms}`             | `X-Requested-With: XMLHttpRequest`   |

`v=v4.X` là hardcoded site version (rev khi site bump build). Nếu trả 400, mở DevTools 1 lần → đọc query → bump.

Search endpoint là SPA-only → không support trong manifest hiện tại (Tachiyomi-style global search là phase sau).

**Pagination**: `/rank/day|week|month` là SSR fixed 100 items, không phân trang. AJAX `/apis/c/index?pn=N` có pagination thật (24/page, `isEnd` flag) NHƯNG bị **Cloudflare gate yêu cầu `cf_clearance` cookie**. Cookie bound với client IP+UA, không thể auto-fetch server-side. Bỏ qua AJAX feed; 3 rank × 100 = ~250 distinct manga đủ cho phase này. User cần thư viện rộng hơn thì pivot sang MangaDex/OTruyen.

---

## 8. Khi mọi cách thất bại

Có site khả năng **không scrape được** trong phase 1:

- CF Turnstile / JS challenge trên mọi pageload — cần Flaresolverr-style headless solver.
- WebSocket-only reader (cực hiếm với manga).
- TLS-fingerprint anti-bot — Cloudflare Worker không bypass được.

→ Bỏ qua, ghi note trong manifest README, làm site khác.
