/** typoon-scan — container gateway worker.
 *
 * ScanService.scanChapter → single container call.
 * Container runs full typoon pipeline: comic_detr + Lens + spatial_join.
 * Builds storyboard in-memory (pages already decoded) and returns storyboard_keys.
 */

import { Container, getRandom } from "@cloudflare/containers";
import { WorkerEntrypoint } from "cloudflare:workers";

interface Env {
  SCAN_CONTAINER:       DurableObjectNamespace<ScanContainer>;
  MAX_INSTANCES?:       string;
  LENS_ENDPOINT?:       string;
  AWS_ACCESS_KEY_ID:    string;
  AWS_SECRET_ACCESS_KEY:string;
  R2_ACCOUNT_ID:        string;
  R2_BUCKET_NAME:       string;
}

export class ScanContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter  = "10m";
  envVars = {
    AWS_ACCESS_KEY_ID:     this.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY: this.env.AWS_SECRET_ACCESS_KEY,
    R2_ACCOUNT_ID:         this.env.R2_ACCOUNT_ID,
    R2_BUCKET_NAME:        this.env.R2_BUCKET_NAME,
    LENS_ENDPOINT:         this.env.LENS_ENDPOINT ?? "",
  };
}

export interface ScanChapterArgs {
  chapter_id:   number;
  workflow_id:  string;
  pages:        { page_index: number; prepared_key: string; is_color: boolean }[];
  lang_hint?:   string;
  total_pages?: number;
}

export interface ScanChapterResult {
  scan_keys:       string[];
  mask_keys:       string[];
  storyboard_keys: string[];
  timings_ms:      Record<string, number>;
}

export class ScanService extends WorkerEntrypoint<Env> {
  async scanChapter(args: ScanChapterArgs): Promise<ScanChapterResult> {
    const max  = parseInt(this.env.MAX_INSTANCES ?? "3", 10);
    const stub = await getRandom(this.env.SCAN_CONTAINER, max);

    const url = new URL("http://container/scan");
    url.searchParams.set("chapter_id", String(args.chapter_id));

    const resp = await stub.containerFetch(url.toString(), {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pages:     args.pages,
        lang_hint: args.lang_hint ?? "",
      }),
    });
    if (!resp.ok) throw new Error(`scan container ${resp.status}: ${await resp.text()}`);
    return resp.json() as Promise<ScanChapterResult>;
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      const stub = await getRandom(env.SCAN_CONTAINER, parseInt(env.MAX_INSTANCES ?? "3", 10));
      return stub.containerFetch("http://container/health");
    }
    if (url.pathname === "/warm") {
      const stub = await getRandom(env.SCAN_CONTAINER, parseInt(env.MAX_INSTANCES ?? "3", 10));
      return stub.containerFetch("http://container/warm");
    }
    if (url.pathname === "/pingproxy") {
      const stub = await getRandom(env.SCAN_CONTAINER, parseInt(env.MAX_INSTANCES ?? "3", 10));
      const u = new URL("http://container/pingproxy");
      u.search = url.search;
      return stub.containerFetch(u.toString());
    }
    if (url.pathname === "/debug-scan" && req.method === "POST") {
      const stub = await getRandom(env.SCAN_CONTAINER, parseInt(env.MAX_INSTANCES ?? "3", 10));
      const u = new URL("http://container/scan");
      u.searchParams.set("chapter_id", url.searchParams.get("chapter_id") ?? "debug");
      return stub.containerFetch(u.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: await req.text(),
      });
    }
    return Response.json({ ok: true, service: "typoon-scan", rpc: "ScanService.scanChapter" });
  },
};
