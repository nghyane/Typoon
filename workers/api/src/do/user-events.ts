/**
 * UserEventsDO — one Durable Object per user, multiplexes job updates
 * across every device + tab that user has open.
 *
 *   /api/me/events?token=…  →  proxied to UserEventsDO.idFromName(String(user_id))
 *
 * Lifecycle:
 *   1. Client opens a WS once per app session (root provider).
 *   2. DO replays a fresh snapshot of the user's jobs (last 7 d) so newly
 *      mounted UI starts with full state, no per-job poll needed.
 *   3. Pipeline callbacks (notifyProgress / finalize / notifyError) update
 *      D1 then call `UserEventsDO.publishJob(job_id)` over RPC, which fans
 *      a fresh `ApiJob` to every connected session.
 *
 * Hibernatable WebSockets → idle = 0 cost. Session count is tracked via
 * `ctx.getWebSockets()` which survives hibernation.
 */

import { DurableObject } from "cloudflare:workers";

import type { Env, ApiJob } from "../types";
import { withSignedUrls } from "../lib/api-job";
import { getJob, listJobsForUser } from "../store/jobs";

export type UserEvent =
  | { kind: "snapshot"; jobs: ApiJob[] }
  | { kind: "job";      job:  ApiJob };

export class UserEventsDO extends DurableObject<Env> {
  override async fetch(req: Request): Promise<Response> {
    if (req.headers.get("Upgrade") !== "websocket") {
      return new Response("Expected WebSocket upgrade", { status: 426 });
    }

    // The caller stuffed user_id into the URL — see `/api/me/events` route.
    const userId = Number(new URL(req.url).searchParams.get("uid"));
    if (!Number.isFinite(userId) || userId <= 0) {
      return new Response("Missing user_id", { status: 400 });
    }

    const pair = new WebSocketPair();
    const [client, server] = [pair[0], pair[1]];

    // Persist user_id alongside the socket so notify() knows who to fetch
    // jobs for on snapshot replay (DO is keyed by user_id, but attachments
    // survive hibernation and avoid one extra lookup per push).
    this.ctx.acceptWebSocket(server);
    server.serializeAttachment({ userId });

    // Send a full snapshot on connect. Clients merge by id, so this also
    // doubles as a "reconnect after disconnect" recovery path.
    const rows = await listJobsForUser(this.env.DB, userId, 50);
    const jobs = await Promise.all(rows.map(r => withSignedUrls(this.env, r)));
    const snapshot: UserEvent = { kind: "snapshot", jobs };
    server.send(JSON.stringify(snapshot));

    return new Response(null, { status: 101, webSocket: client });
  }

  /** Called by PipelineCallback after every D1 mutation. Pulls the latest
   *  row, builds a signed-URL `ApiJob`, broadcasts to all sessions. */
  async publishJob(job_id: number): Promise<void> {
    const row = await getJob(this.env.DB, job_id);
    if (!row) return;
    const job = await withSignedUrls(this.env, row);
    const event: UserEvent = { kind: "job", job };
    const msg = JSON.stringify(event);
    for (const ws of this.ctx.getWebSockets()) {
      try { ws.send(msg); }
      catch { /* dead session; runtime cleans up */ }
    }
  }

  override webSocketClose(ws: WebSocket, code: number): void {
    ws.close(code);
  }

  override webSocketError(_ws: WebSocket, _err: unknown): void {
    /* runtime handles cleanup */
  }
}
