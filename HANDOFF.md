# Typoon v2 — Handoff

## Trạng thái hiện tại

Branch: `main` (uncommitted changes từ session này)
CWD: `/Users/nghiahoang/Dev/MANGA/ComicScan/v2`
Config: `~/.typoon/config.toml`

### Projects
- `system-universe` (en→vi): ch001 ✓, ch006 ✓, ch003 đang stuck ở `translating`
- `sample_chapters` (ko→vi): ch001 ✓, ch002 ✓

### Config hiện tại
```toml
[translation]
provider = "local"
model = "gpt-4.1-mini"
max_tokens = 8192

[context_agent]
provider = "local"
model = "auto"       # routes tới gpt-5-mini qua LunchDock proxy
max_tokens = 16384

[vision_agent]
provider = "local"
model = "auto"
```

Local proxy: `http://localhost:8080/v1` — LunchDock/Bifrost, routes `auto` → `gpt-5-mini` (GitHub Copilot)
MiniMax provider cũng có sẵn nếu proxy down.

---

## Kiến trúc sau refactor (session này)

### Pipeline flow
```
CLI → ProjectService._run_chapter(hook)
    → pipeline.run(cp, ctx, runtime, hook=hook)
        → _run_prepare / _run_scan / _run_translate / _run_render
            → translate_chapter(scanned, ctx)
                → build_chapter_brief(ctx, prepared, keyed)   # context agent
                → asyncio.gather(translate_window × N)         # parallel windows
```

### Key types
- `TranslateCtx` (`adapters/ctx.py`) — thay Session, frozen dataclass, explicit providers
- `BubbleKey` (`domain/scan.py`) — stable identity: `key` (hash) + `bubble`
- `tool_loop` (`llm/loop.py`) — single LLM loop, parallel tool dispatch

### Removed
- `Session`, `make_session`, `agent.py`, `Agent` protocol
- Streaming infrastructure (`StreamEvent`, `StreamEventType`, `Provider.stream()`)
- Async generator pipeline

---

## Vấn đề chưa giải quyết

### 1. ch003 stuck ở `translating`
Reset và chạy lại:
```bash
typoon translate system-universe --from 3 --to 3 --redo translate
```

Root cause lần cuối: context agent (model `auto`) gọi `look_at` nhiều lần → chậm.
Nếu proxy down, đổi context sang minimax:
```toml
[context_agent]
provider = "minimax"
model = "MiniMax-M2.7"
```

### 2. Context agent gọi look_at quá nhiều
Prompt đã được siết (`ONLY when ALL of these are true`), nhưng model vẫn có thể spam.
Nếu vẫn xảy ra: bỏ `look_at` khỏi tools list trong `context.py:tools` để disable hoàn toàn.

### 3. Rate limit 429 khi parallel windows
Proxy LunchDock có rate limit. Nếu gặp 429, thêm semaphore vào `translate.py`:
```python
sem = asyncio.Semaphore(2)
async def _w(i, wk):
    async with sem:
        return await translate_window(...)
results = await asyncio.gather(*[_w(i, wk) for i, wk in enumerate(windows)])
```

### 4. Xưng hô vẫn có thể sai
Brief build đúng (address rules có đủ), nhưng page agent không biết ai đang nói.
`key_notes` trong brief có thể annotate speaker per bubble — context agent cần làm điều này.
Verify bằng cách xem brief của chapter: `typoon status` rồi query DB.

---

## Cách chạy E2E

```bash
cd /Users/nghiahoang/Dev/MANGA/ComicScan/v2

# Kiểm tra proxy còn sống không
curl -s http://localhost:8080/v1/models | python3 -m json.tool | grep '"id"'

# Dịch chapter
NO_COLOR=1 typoon translate system-universe --from 3 --to 3 --redo translate 2>&1 | tee /tmp/run.log

# Export PDF
cd ~/.typoon/projects/system-universe/ch003/render
python3 -c "
from PIL import Image; import glob
imgs = [Image.open(p).convert('RGB') for p in sorted(glob.glob('page_*.png'))]
imgs[0].save('/Users/nghiahoang/Desktop/ch003.pdf', save_all=True, append_images=imgs[1:])
print(len(imgs), 'pages')
"
```

---

## Files quan trọng đã thay đổi (uncommitted)

```
typoon/llm/loop.py          # NEW — tool_loop, parallel dispatch
typoon/llm/ir.py            # removed streaming
typoon/llm/openai.py        # removed stream(), timeout=300
typoon/llm/anthropic.py     # non-streaming create()
typoon/llm/conversation.py  # NEW — ConversationBuffer
typoon/adapters/ctx.py      # NEW — TranslateCtx, make_ctx
typoon/adapters/project_service.py  # uses make_ctx, plain pipeline.run
typoon/adapters/session.py  # DELETED
typoon/stages/pipeline.py   # plain async def, no generator
typoon/stages/translate.py  # uses TranslateCtx
typoon/agents/context.py    # plain function, uses tool_loop
typoon/agents/page.py       # plain function, uses ConversationBuffer
typoon/agents/look_at.py    # uses TranslateCtx, max 3 pages
typoon/agents/brief.py      # uses list[BubbleKey]
typoon/agents/keys.py       # returns list[BubbleKey]
typoon/domain/scan.py       # added BubbleKey
typoon/domain/translate.py  # added to_records()
typoon/storage/records.py   # simplified BriefRecord
typoon/storage/sqlite.py    # save_translations accepts list[dict]
typoon/cli/events.py        # full traceback, ToolResult log
```

---

## Kinh nghiệm

**Proxy LunchDock (localhost:8080)**
- Model `auto` → routes tới `gpt-5-mini` (GitHub Copilot)
- Rate limit: ~15 req/min, 429 khi parallel quá nhiều
- Timeout server-side: 30s — context agent với nhiều tool calls dễ timeout
- Model `gpt-4.1-mini` explicit cũng hoạt động

**MiniMax (api.minimax.io/anthropic)**
- Dùng Anthropic SDK nhưng non-streaming (SDK accumulator bug)
- Tốt cho context agent vì context window lớn hơn
- Prompt caching hoạt động (cache_control trên system + tools)

**CF Gemma (workers-ai)**
- Không hỗ trợ tool calls tốt — context agent thường không submit brief
- Dùng được cho translation (text-only, XML output)
- Vision: chậm (~9s/page), không nên dùng cho look_at nhiều pages

**Xưng hô tiếng Việt**
- Brief có `address` rules đúng nhưng page agent không biết speaker
- `key_notes` là nơi đúng để annotate speaker per bubble
- Context agent cần được prompt rõ hơn để fill key_notes với speaker info
