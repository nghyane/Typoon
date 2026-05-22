/** typoon-inpaint — thin gateway worker.
 *
 *   InpaintService.inpaintChapter  → POST /inpaint-chapter on the container.
 *                                    Container reads R2 directly (aws4 S3
 *                                    client), processes N pages in parallel
 *                                    (saturates 4 vCPU × standard-4), and
 *                                    writes inpaint/{j}/{i}.png for each.
 *
 *   InpaintContainer (DO)          → the actual container, managed by
 *                                    @cloudflare/containers. Per-page work
 *                                    + chapter batch live in
 *                                    crates/inpaint/src/page.rs.
 *
 * Per-page decode + mask close + tile build + inpaint + compose + PNG
 * encode all happen in the container — the Worker isolate's 128 MB heap
 * couldn't fit a single 2000×3771 RGBA decode + Candle tensors. Batching
 * the whole chapter into one container call avoids the per-page hop
 * (~5 ms × N) and lets the container saturate its full vCPU budget.
 */

import { WorkerEntrypoint } from "cloudflare:workers";
import { Container, getRandom } from "@cloudflare/containers";

interface Env {
  INPAINT_CONTAINER:    DurableObjectNamespace<InpaintContainer>;
  MAX_INSTANCES?:       string;
  AWS_ACCESS_KEY_ID:    string;
  AWS_SECRET_ACCESS_KEY:string;
  R2_ACCOUNT_ID:        string;
  R2_BUCKET_NAME:       string;
}

export class InpaintContainer extends Container<Env> {
  defaultPort = 8000;
  // Hold the warm Candle session for a full chapter (~1 min at p95). Pages
  // arrive batched, so a chapter keeps the container hot end-to-end; only
  // multi-chapter idle gaps see a cold reload.
  sleepAfter  = "5m";
  envVars = {
    AWS_ACCESS_KEY_ID:     this.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY: this.env.AWS_SECRET_ACCESS_KEY,
    R2_ACCOUNT_ID:         this.env.R2_ACCOUNT_ID,
    R2_BUCKET_NAME:        this.env.R2_BUCKET_NAME,
  };
}

export interface InpaintChapterArgs {
  job_id:       number;
  page_indices: number[];
  /** In-instance concurrency. Default 4 — matches the standard-4 vCPU
   *  count, which probe data shows is where Candle CPU saturates. */
  concurrency?: number;
}

export interface InpaintPageOutcome {
  page_index:  number;
  output_key?: string;
  bubbles?:    number;
  tiles_shape?: string[];
  error?:      string;
}

export interface InpaintChapterResult {
  results:          InpaintPageOutcome[];
  wall_total_ms:    number;
  concurrency_used: number;
}

export class InpaintService extends WorkerEntrypoint<Env> {
  /** Inpaint an entire chapter in one container call. The workflow shards
   *  large chapters across instances by submitting one chapter request per
   *  container instance (round-robin via getRandom). Returns a per-page
   *  outcome list so the workflow knows exactly which pages failed. */
  async inpaintChapter(args: InpaintChapterArgs): Promise<InpaintChapterResult> {
    const max  = parseInt(this.env.MAX_INSTANCES ?? "3", 10);
    const stub = await getRandom(this.env.INPAINT_CONTAINER, max);

    const resp = await stub.containerFetch("http://container/inpaint-chapter", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(args),
    });
    if (!resp.ok) {
      throw new Error(`inpaint container ${resp.status}: ${await resp.text()}`);
    }
    return resp.json() as Promise<InpaintChapterResult>;
  }

  async warm(): Promise<{ ok: true }> {
    const max  = parseInt(this.env.MAX_INSTANCES ?? "3", 10);
    const stub = await getRandom(this.env.INPAINT_CONTAINER, max);
    const resp = await stub.containerFetch("http://container/health");
    if (!resp.ok) throw new Error(`warm: container ${resp.status}`);
    return { ok: true };
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      const max  = parseInt(env.MAX_INSTANCES ?? "3", 10);
      const stub = await getRandom(env.INPAINT_CONTAINER, max);
      return stub.containerFetch("http://container/health");
    }
    // Diagnostic passthrough — issues an inpaint pass with full per-stage
    // timings against a real (job_id, page_index) that already has
    // prepared/mask/scan in R2. Writes a probe key so production
    // artifacts aren't clobbered. Pin to one instance (`bench` DO name)
    // so repeat runs reuse the same warm container.
    //
    // Path is `/bench`, not `/probe` or `/diag`: Cloudflare's edge
    // intercepts POST to those paths (likely reserved for platform
    // health checks) and returns 404 before our worker fetch runs.
    if (url.pathname === "/bench" && req.method === "POST") {
      const id   = env.INPAINT_CONTAINER.idFromName("bench");
      const stub = env.INPAINT_CONTAINER.get(id);
      const inner = new URL("http://container/bench");
      for (const [k, v] of url.searchParams) inner.searchParams.set(k, v);
      return stub.containerFetch(inner.toString(), { method: "POST" });
    }
    return Response.json({
      ok:      true,
      service: "typoon-inpaint",
      rpc:     "InpaintService.inpaintChapter",
    });
  },
};
