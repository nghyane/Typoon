/** typoon-inpaint — Worker entry: Container gateway + Queue consumer + legacy RPC.
 *
 *   queue(batch)                — Queue consumer (max_concurrency=10)
 *                                 triggered by pipeline.enqueue-inpaints.
 *                                 Decrements FanInCounterDO; last one home
 *                                 wakes the workflow via PipelineNotifier.
 *
 *   InpaintService.inpaintPage  — Legacy RPC for ad-hoc/probe callers.
 *
 *   InpaintContainer (DO)       — The actual container (managed by
 *                                 @cloudflare/containers). Tile inference
 *                                 lives in container/main.py. */

import { WorkerEntrypoint } from "cloudflare:workers";

// WASM modules — Wrangler resolves .wasm imports to WebAssembly.Module at runtime.
import __JPEG_DEC_WASM__ from "../../node_modules/@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm";
import __PNG_WASM__      from "../../node_modules/@jsquash/png/codec/pkg/squoosh_png_bg.wasm";
(globalThis as any).__JPEG_DEC_WASM__ = __JPEG_DEC_WASM__;
(globalThis as any).__PNG_WASM__      = __PNG_WASM__;

import { runInpaintPage, type InpaintPageArgs, type InpaintPageResult } from "./inpaint";
import {
  InpaintContainer, inpaintTile, warmContainer, type ContainerEnv,
} from "./container";

export { InpaintContainer };

// ── Message shape (must match pipeline/src/types.ts InpaintJob) ───────────────

interface InpaintJob {
  workflow_id:  string;
  chapter_id:   string;
  page_index:   number;
  total_pages:  number;
  prepared_key: string;
  scan_key:     string;
  mask_key:     string;
}

// ── Fan-in DO + notifier (cross-worker bindings) ──────────────────────────────

interface FanInStub {
  decrement(workflowId: string, stage: string): Promise<boolean>;
}
interface PipelineNotifierStub extends Fetcher {
  notify(workflowId: string, eventType: string, payload: unknown): Promise<void>;
}

interface Env extends ContainerEnv {
  R2:              R2Bucket;
  FAN_IN:          DurableObjectNamespace<FanInStub>;
  PIPELINE_WORKER: PipelineNotifierStub;
}

// ── Key helpers (mirror pipeline K) ───────────────────────────────────────────

const pad = (i: number) => String(i).padStart(4, "0");
const inpaintKey = (c: string, i: number) => `inpaint/${c}/${pad(i)}.png`;

// ── ExecutionContext adapter for queue consumer ───────────────────────────────
//
// runInpaintPage takes an ExecutionContext (for the R2 write-through cache's
// waitUntil). Queue handlers receive the ctx too — we pass it through.

function makeStub(env: Env) {
  return {
    inpaintTile: (body: Uint8Array, W: number, H: number) =>
      inpaintTile(env, body, W, H),
  };
}

// ── Queue consumer + liveness ─────────────────────────────────────────────────

export default {
  async fetch(_req: Request, env: Env): Promise<Response> {
    const url = new URL(_req.url);
    if (url.pathname === "/health") {
      const t = await warmContainer(env);
      return Response.json({ ok: true, container: t });
    }
    return Response.json({
      ok:      true,
      service: "typoon-inpaint",
      rpc:     "InpaintService.inpaintPage",
      consumer:"typoon-inpaint-queue",
    });
  },

  /** Queue consumer. max_batch_size=1 → one page per invocation; isolates
   *  don't share memory so concurrent invocations are safe. */
  async queue(batch: MessageBatch<InpaintJob>, env: Env, ctx: ExecutionContext): Promise<void> {
    for (const msg of batch.messages) {
      try {
        await processOneInpaint(msg.body, env, ctx);
        msg.ack();
      } catch (e) {
        console.error("inpaint failed", msg.body.chapter_id, msg.body.page_index, e);
        msg.retry();
      }
    }
  },
};

async function processOneInpaint(
  job: InpaintJob, env: Env, ctx: ExecutionContext,
): Promise<void> {
  await runInpaintPage(
    {
      chapter_id: job.chapter_id,
      page_index: job.page_index,
      image_key:  job.prepared_key,
      mask_key:   job.mask_key,
      scan_key:   job.scan_key,
    },
    { R2: env.R2, INPAINT: makeStub(env) },
    ctx,
  );

  // Atomic decrement; last consumer notifies the workflow.
  const fanIn = env.FAN_IN.getByName(job.workflow_id);
  const last  = await fanIn.decrement(job.workflow_id, "inpaints-done");
  if (!last) return;

  const inpaint_keys = Array.from(
    { length: job.total_pages },
    (_, i) => inpaintKey(job.chapter_id, i),
  );
  await env.PIPELINE_WORKER.notify(job.workflow_id, "inpaints-done", { inpaint_keys });
}

// ── Legacy RPC (kept for ad-hoc / probe callers) ──────────────────────────────

export class InpaintService extends WorkerEntrypoint<Env> {
  async inpaintPage(args: InpaintPageArgs): Promise<InpaintPageResult> {
    return runInpaintPage(
      args,
      { R2: this.env.R2, INPAINT: makeStub(this.env) },
      this.ctx,
    );
  }
  async warm(): Promise<{ ok: true; timings: Record<string, number> }> {
    const t = await warmContainer(this.env);
    return { ok: true, timings: t };
  }
}
