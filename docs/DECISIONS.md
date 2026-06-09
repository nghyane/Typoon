# Design decisions

One entry per non-obvious choice. Reference these from commits and
review comments instead of re-litigating.

## D1. Cloudflare edge + Fly.io CPU compute

**Choice**: API + workflow + LLM stages on Cloudflare Workers; image
compute (prepare, scan, render-pack) on Fly.io machines with
`performance-4x` (4 vCPU × 8 GB).

**Rejected**:

- **Cloudflare Containers**: $12/1000ch vs Fly $5/1000ch; image size
  limits; container DO billing per RAM-sec even idle.
- **AWS Lambda**: 10 GB image limit doable but cold-start + Python
  cold imports slow.
- **Modal**: $27/1000ch, vendor lock, less control over Dockerfile.
- **Self-host Hetzner / Mac Mini now**: defer until > 5K chapter/month
  proves the volume.
- **Vast.ai Serverless GPU**: GPU only cuts ~40% wall (CPU pre/post is
  the floor), marketplace reliability + PyWorker template lock-in not
  worth $2/1000ch saving at < 5K chapter/month.

**Trade-off**: two vendors instead of one; one `COMPUTE_URL` worker
secret bridges them. Migration target (Hetzner / Mac Mini) keeps the
same image — only the URL changes.

## D2. Two RPC endpoints, not one monolith

**Choice**: Fly exposes `POST /prepare-scan` and `POST /render-pack`.
Workflow calls them as two `step.do`.

**Rejected**:

- **One `/process-chapter`**: container would idle ~38 s waiting on
  LLM stages — wasted RAM-sec.
- **One `/process-chapter` with LLM inside container**: loses workflow
  checkpoints (one LLM transient = re-run prepare+scan from zero),
  forces LLM API keys onto Fly.

**Trade-off**: two cold-starts per chapter instead of one. At ~3 s
each + Fly auto-start, the extra ~$0.0002/chapter is negligible. The
gain is workflow checkpointing between brief / translate and
render-pack.

## D3. Two inpaint tiers only

**Choice**: `InpaintTier.FLAT_FILL = 0`, `InpaintTier.AOT_GAN = 1`.

**Rejected**:

- **Tier 2 (Telea / Navier-Stokes via OpenCV)**: +50 MB image weight
  for marginal quality on the ~10% of bubbles that sit between
  flat-color and complex art.
- **SSIM auto-escalation**: adds runtime overhead and a feedback loop
  that's hard to reason about. Heuristic + AOT-GAN fallback is enough.

**Trade-off**: ~10% of bubbles that could be handled by a cheap
non-neural inpaint instead go through AOT-GAN. Slight cost increase
in exchange for simpler dispatch.

## D4. Minimal `ChapterImageManifest`

**Choice**: 5 fields — `job_id`, `strategy`, `is_color`,
`source_lang_hint`, `pages[{ index, width, height, source[] }]`.

**Rejected fields**:

- `prepared_key` — derivable from convention via
  `preparedKey(job_id, index)`.
- `sha` — no stage consumes content hash.
- `page_kind[]` — belongs in `InpaintPlan` (computed in scan, embedded
  in msgpack).
- `prepared_at` — workflow event already has timestamp + instance id.
- `raw_count` — derivable from `sum(pages[i].source.length)`.

## D5. Coordinate space contract

**Choice**: every persisted artifact uses **`prepared_space`** —
pixel coordinates of `prepared/{job}/{i:04d}.jpg`.

`lens_space` (1100 px Lens upload) exists only inside
`LensBlocksDetector.detect()`. Coordinates are scaled back before
returning.

`tile_space` (AOT-GAN 128/192/256/384 buckets) exists only inside the
inpaint runtime per tile.

**Rejected**: per-stage coordinate transforms. Causes drift between
mask, scan polygons, and render target.

## D6. Job-affinity Fly machine

**Choice**: Cloudflare worker pins both `/prepare-scan` and
`/render-pack` calls of the same chapter to the same Fly machine by
including `?job_id=N` in the URL (Fly's machine routing is best-effort
but the same caller usually lands on the same warm machine within
the `auto_stop` window).

If Fly fly-replay or per-machine pin proves necessary later, switch
to `fly-replay` headers or one-machine-per-chapter via the Machines
API.

**Rationale**: warm AOT model load (~5 s) is paid once per chapter
across two calls; warm OS page cache for R2 reads.

**Trade-off**: load is not perfectly balanced. Acceptable at
20 concurrent.

## D7. Storyboard for brief LLM

**Choice**: keep `storyboard/{job}/{n:02d}.jpg` composite with key
labels. Tune `_CANVAS_MAX_EDGE` from 2048 → 1536 px and
`STORYBOARD_JPEG_Q` from 82 → 78.

**Rationale**: brief vision LLM needs labeled bubble overlay to map
text↔position. Smaller storyboard ⇒ fewer vision tokens.

**Risk**: too small storyboard may make labels unreadable for the
LLM. Probe before locking the new values.

## D8. Lens resize 1500 → 1100 px

**Choice**: in `LensBlocksDetector._encode_for_lens`, set
`MAX_DIM = 1100`.

**Rationale**: Lens server downsamples to ~1000 px regardless;
sending bytes above that wastes upload time.

**Risk**: JA tategaki at low resolution may lose word accuracy.
Monitor `groups[].confidence` distribution after deploy; revert if
drop > 5 %.

## D9. No streaming scan → render

**Choice**: workflow gates on full scan before starting render.

**Rejected**: Queue + FanInCounterDO + PipelineNotifier streaming
pattern (in git history at commit 650dea1).

**Rationale**: at 115 s wall under the 120 s target, the ~20 s
streaming would save is not worth the cross-worker coordination
complexity. Revisit only if target tightens.

## D10. AOT-GAN model baked into image

**Choice**: `COPY fly/aot-gan.safetensors /app/models/aot-gan.safetensors`
in Dockerfile.

**Rejected**:

- **Download from R2 at startup**: adds ~5 s cold start per machine
  × N machines.
- **Mount via Fly Volume**: ties machine to specific host, breaks
  auto-scale.

**Trade-off**: image size ~+50 MB. Acceptable.

## D11. comic-detr ONNX baked into image

Same reasoning as D10. File is ~12 MB.

## D12. R2 lifecycle rules

**Choice**:

```
raw/        7d
prepared/   7d
scan/       30d
storyboard/ 7d
brief/      keep
ctx/        keep
translate/  90d
render/     keep
```

**Rationale**: only `render/` is the user-visible artifact; `brief/`
and `ctx/` enable cross-chapter context reuse. Everything else is
intermediate.

## D13. No inpaint or typeset intermediate files

**Choice**: `/render-pack` keeps RGBA in memory through erase →
text → JPEG encode → BNL pack. No `inpaint/{i}.png` written.

**Rationale**: cuts 100 PNG encode + 100 R2 writes + 100 R2 reads +
100 PNG decodes per chapter ≈ 70 s of saved work + zero intermediate
storage cost.

**Trade-off**: cannot re-render text without re-running erase. We
accept that — re-render is rare and erasing is cheap (~10 s for
a 100-page chapter).
