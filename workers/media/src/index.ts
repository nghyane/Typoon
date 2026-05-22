/** media-container gateway worker.
 *
 * Two RPC methods routed to a Container running Pillow (Python):
 *   MediaService.prepareChapter → Container POST /prepare
 *   MediaService.packChapter    → Container POST /pack
 *
 * Container reads/writes R2 directly via tigrisfs FUSE mount.
 * No R2 binding or proxy route needed in the Worker.
 */

import { Container, getRandom } from "@cloudflare/containers";
import { WorkerEntrypoint } from "cloudflare:workers";
import type { PreparedJobMeta } from "@typoon/shared";

interface Env {
  MEDIA_CONTAINER:      DurableObjectNamespace<MediaContainer>;
  MAX_INSTANCES?:       string;
  // R2 credentials forwarded to container via envVars
  AWS_ACCESS_KEY_ID:    string;
  AWS_SECRET_ACCESS_KEY:string;
  R2_ACCOUNT_ID:        string;
  R2_BUCKET_NAME:       string;
}

export class MediaContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter  = "5m";
  envVars = {
    AWS_ACCESS_KEY_ID:     this.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY: this.env.AWS_SECRET_ACCESS_KEY,
    R2_ACCOUNT_ID:         this.env.R2_ACCOUNT_ID,
    R2_BUCKET_NAME:        this.env.R2_BUCKET_NAME,
  };
}

export class MediaService extends WorkerEntrypoint<Env> {
  private async stub() {
    const max = parseInt(this.env.MAX_INSTANCES ?? "2", 10);
    return getRandom(this.env.MEDIA_CONTAINER, max);
  }

  async prepareChapter(args: {
    job_id: number;
    zip_key:    string;
    strategy?:  "auto" | "one_to_one" | "stitch";
  }): Promise<PreparedJobMeta> {
    const stub = await this.stub();

    // Container reads ZIP directly from R2 via FUSE mount.
    // zip_key is passed so container knows the R2 path.
    const url = new URL("http://container/prepare");
    url.searchParams.set("job_id", String(args.job_id));
    url.searchParams.set("strategy",   args.strategy ?? "auto");
    url.searchParams.set("zip_key",    args.zip_key);

    const resp = await stub.containerFetch(url.toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body:    "{}",
    });
    if (!resp.ok) throw new Error(`media /prepare ${resp.status}: ${await resp.text()}`);
    return resp.json() as Promise<PreparedJobMeta>;
  }

  async buildStoryboard(args: {
    job_id: number;
    pages:      { index: number; width: number; height: number }[];
  }): Promise<{ storyboard_keys: string[] }> {
    const stub = await this.stub();
    const resp = await stub.containerFetch("http://container/storyboard", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(args),
    });
    if (!resp.ok) throw new Error(`media /storyboard ${resp.status}: ${await resp.text()}`);
    return resp.json() as Promise<{ storyboard_keys: string[] }>;
  }

  async packChapter(args: {
    job_id:  number;
    page_keys:   string[];
    output_key?: string;
  }): Promise<{ output_key: string; size_bytes: number; pages: number }> {
    const stub = await this.stub();

    const resp = await stub.containerFetch("http://container/pack", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id:   args.job_id,
        typeset_keys: args.page_keys,
      }),
    });
    if (!resp.ok) throw new Error(`media /pack ${resp.status}: ${await resp.text()}`);
    const result = await resp.json() as { bnl_key: string; size_bytes: number; pages: number };
    return { output_key: result.bnl_key, size_bytes: result.size_bytes, pages: result.pages };
  }

  async warm(): Promise<{ ok: true }> {
    const stub = await this.stub();
    await stub.containerFetch("http://container/health");
    return { ok: true };
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);

    if (url.pathname === "/health") {
      const max  = parseInt(env.MAX_INSTANCES ?? "2", 10);
      const stub = await getRandom(env.MEDIA_CONTAINER, max);
      return stub.containerFetch("http://container/health");
    }

    // Debug: POST /debug-storyboard → forward to container, return full error
    if (url.pathname === "/debug-storyboard" && req.method === "POST") {
      const max  = parseInt(env.MAX_INSTANCES ?? "2", 10);
      const stub = await getRandom(env.MEDIA_CONTAINER, max);
      const body = await req.text();
      const resp = await stub.containerFetch("http://container/storyboard", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      const text = await resp.text();
      return new Response(text, { status: resp.status, headers: { "Content-Type": "application/json" } });
    }

    // Debug: POST /debug-prepare?job_id=&zip_key=&strategy= → return full
    // container stdout + body so we can see Python tracebacks without
    // chasing wrangler tail.
    if (url.pathname === "/debug-prepare" && req.method === "POST") {
      const max  = parseInt(env.MAX_INSTANCES ?? "2", 10);
      const stub = await getRandom(env.MEDIA_CONTAINER, max);
      const inner = new URL("http://container/prepare");
      for (const [k, v] of url.searchParams) inner.searchParams.set(k, v);
      const resp = await stub.containerFetch(inner.toString(), {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    "{}",
      });
      const text = await resp.text();
      return new Response(text, { status: resp.status, headers: { "Content-Type": "application/json" } });
    }

    return Response.json({
      ok:      true,
      service: "typoon-media",
      rpc:     "MediaService.prepareChapter | MediaService.packChapter",
    });
  },
};
