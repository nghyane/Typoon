/** Container gateway — wraps the Python tile-inference service.
 *
 * Exposes a single async function `inpaintTile(body, W, H)` that other
 * modules use as a black-box tile inference. The container itself is a
 * Cloudflare Container running ONNX Runtime (see container/main.py). */

import { Container, getRandom } from "@cloudflare/containers";

export class InpaintContainer extends Container<unknown> {
  defaultPort = 8080;
  // Keep the warm Python ORT session alive across a whole chapter's pages.
  // Cold start ~5s; subsequent tiles ride the warm session.
  sleepAfter  = "5m";
}

export interface ContainerEnv {
  INPAINT_CONTAINER: DurableObjectNamespace<InpaintContainer>;
  MAX_INSTANCES?:    string;
}

/** Pick a random container instance and run one tile through it. */
export async function inpaintTile(
  env: ContainerEnv, body: Uint8Array, W: number, H: number,
): Promise<Uint8Array> {
  const max  = parseInt(env.MAX_INSTANCES ?? "3", 10);
  const stub = await getRandom(env.INPAINT_CONTAINER, max);

  const resp = await stub.containerFetch(`http://container/inpaint?w=${W}&h=${H}`, {
    method:  "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body,
  });
  if (!resp.ok) {
    throw new Error(`inpaint container ${resp.status}: ${await resp.text()}`);
  }
  return new Uint8Array(await resp.arrayBuffer());
}

/** Pre-load the container's ORT session without running inference. */
export async function warmContainer(env: ContainerEnv): Promise<{ load_ms: number }> {
  const max  = parseInt(env.MAX_INSTANCES ?? "3", 10);
  const stub = await getRandom(env.INPAINT_CONTAINER, max);
  const resp = await stub.containerFetch("http://container/health");
  if (!resp.ok) throw new Error(`warm: container ${resp.status}`);
  const body = await resp.json() as { load_ms?: number };
  return { load_ms: body.load_ms ?? -1 };
}
