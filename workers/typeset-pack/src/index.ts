/** typeset-pack gateway worker.
 *
 * TypesetPackService.typesetAndPack:
 *   Container reads inpaint PNGs + scan msgpacks via FUSE
 *   → WASM render_page per page (Node.js subprocess)
 *   → Pillow JPEG encode → BNL pack
 *   → write render/{chapter}.bnl via FUSE
 */

import { Container, getRandom } from "@cloudflare/containers";
import { WorkerEntrypoint } from "cloudflare:workers";

interface Env {
  TYPESET_PACK_CONTAINER: DurableObjectNamespace<TypesetPackContainer>;
  MAX_INSTANCES?:         string;
  AWS_ACCESS_KEY_ID:      string;
  AWS_SECRET_ACCESS_KEY:  string;
  R2_ACCOUNT_ID:          string;
  R2_BUCKET_NAME:         string;
}

export class TypesetPackContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter  = "5m";
  envVars = {
    AWS_ACCESS_KEY_ID:     this.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY: this.env.AWS_SECRET_ACCESS_KEY,
    R2_ACCOUNT_ID:         this.env.R2_ACCOUNT_ID,
    R2_BUCKET_NAME:        this.env.R2_BUCKET_NAME,
  };
}

export interface TypesetPackArgs {
  job_id:    number;
  pages:         { page_index: number; inpaint_key: string; scan_key: string; page_width: number }[];
  translate_key: string;
}

export interface TypesetPackResult {
  archive_key: string;
  size_bytes:  number;
  pages:       number;
  timings_ms:  Record<string, number>;
}

export class TypesetPackService extends WorkerEntrypoint<Env> {
  async typesetAndPack(args: TypesetPackArgs): Promise<TypesetPackResult> {
    const max  = parseInt(this.env.MAX_INSTANCES ?? "3", 10);
    const stub = await getRandom(this.env.TYPESET_PACK_CONTAINER, max);

    const resp = await stub.containerFetch("http://container/typeset-pack", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(args),
    });
    if (!resp.ok) throw new Error(`typeset-pack ${resp.status}: ${await resp.text()}`);
    return resp.json() as Promise<TypesetPackResult>;
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      const stub = await getRandom(env.TYPESET_PACK_CONTAINER, parseInt(env.MAX_INSTANCES ?? "3", 10));
      return stub.containerFetch("http://container/health");
    }
    return Response.json({ ok: true, service: "typoon-typeset-pack" });
  },
};
